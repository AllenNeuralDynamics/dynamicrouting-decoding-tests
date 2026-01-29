import datetime
import os
import random
os.environ['RUST_BACKTRACE'] = '1'
#os.environ['POLARS_MAX_THREADS'] = '1'
os.environ['TOKIO_WORKER_THREADS'] = '1' 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['RAYON_NUM_THREADS'] = '1'

import concurrent.futures as cf
import contextlib
import itertools
import logging
import math
import multiprocessing
import random
import uuid
from typing import Annotated, Iterable, Literal, Sequence

import numpy as np
import polars as pl
import polars._typing
import pydantic_settings
import pydantic_settings.sources
import pydantic.functional_serializers
import tqdm
import upath
from dynamic_routing_analysis.decoding_utils import decoder_helper, NotEnoughBlocksError
from dynamic_routing_analysis import data_utils

import utils

logger = logging.getLogger(__name__)

# define run params here ------------------------------------------- #
Expr = Annotated[
    pl.Expr, pydantic.functional_serializers.PlainSerializer(lambda expr: expr.meta.serialize(format='json'), return_type=str)
]
class BinnedRelativeIntervalConfig(pydantic.BaseModel):
    event_column_name: str
    start_time: float 
    stop_time: float
    bin_size: float

    @property
    def intervals(self) -> list[tuple[float, float]]:
        start_times = np.arange(self.start_time, self.stop_time, self.bin_size)
        stop_times = start_times + self.bin_size
        return list(zip(start_times, stop_times))
    
def to_polars_expr(value: str | pl.Expr) -> Expr:
    if isinstance(value, pl.Expr):
        return value
    """Eval str to create pl.Expr instance"""
    if "pl." not in value:
        raise ValueError(f"Polars expression must access Polars objects under the `pl.` namespace {value=}")
    return eval(value)

class Params(pydantic_settings.BaseSettings):
    # ----------------------------------------------------------------------------------
    # Required parameters
    result_prefix: str
    "An identifier for the decoding run, used to name the output files (can have duplicates with different run_id)"
    # ----------------------------------------------------------------------------------
    
    # Capsule-specific parameters -------------------------------------- #
    session_id: str | None = pydantic.Field(None, exclude=True, repr=True)
    """If provided, only process this session_id. Otherwise, process all sessions that match the filtering criteria"""
    run_id: str = pydantic.Field(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) # created at runtime: same for all Params instances 
    """A unique string that should be attached to all decoding runs in the same batch"""
    skip_existing: bool = pydantic.Field(True, exclude=True, repr=True)
    test: bool = pydantic.Field(False, exclude=True)
    logging_level: str | int = pydantic.Field('INFO', exclude=True)
    update_packages_from_source: bool = pydantic.Field(False, exclude=True)
    override_params_json: str | None = pydantic.Field('{}', exclude=True)
    use_process_pool: bool = pydantic.Field(True, exclude=True, repr=True)
    max_workers: int | None = pydantic.Field(int(os.environ['CO_CPUS']), exclude=True, repr=True)
    """For process pool"""

    # Decoding parameters ----------------------------------------------- #
    session_table_query: str = "is_ephys & is_task & is_annotated & is_production & issues=='[]'"
    unit_criteria: str = pydantic.Field("loose_drift", exclude=True, repr=True) # often varied, stored in data not params file
    unit_subsample_size: int | None = pydantic.Field(None, exclude=True, repr=True) # often varied, so will be stored with data, not in the params file
    """number of units to sample for each area"""
    n_repeats: int = 25
    """number of times to repeat decoding with different randomly sampled units"""
    min_n_units: int = 20
    """only process areas with at least this many units"""
    input_data_type: Literal['spikes', 'facemap', 'LP'] = 'spikes'
    spikes_time_before: float = pydantic.Field(0.2, deprecated="Use time_interval_config instead")
    crossval: Literal['5_fold', 'blockwise', '5_fold_set_random_state'] = '5_fold'
    """blockwise untested with linear shift"""
    labels_as_index: bool = True
    """convert labels (context names) to index [0,1]"""
    decoder_type: Literal['linearSVC', 'nonlinearSVC', 'LDA', 'RandomForest', 'LogisticRegression'] = 'LogisticRegression'
    regularization: float | None = None
    """ set regularization (C) for the decoder. Setting to None reverts to the default value (usually 1.0) """
    penalty: str | None = None
    """ set penalty for the decoder. Setting to None reverts to default """
    solver: str | None = None
    """ set solver for the decoder. Setting to None reverts to default """
    units_group_by: list[str] = ['session_id', 'structure', 'electrode_group_names']

    trials_filter: Annotated[str | Expr, pydantic.AfterValidator(to_polars_expr)] = pydantic.Field(default_factory = lambda:pl.lit(True))
    """ filter trials table input to decoder by boolean column or polars expression"""
    label_to_decode: str = 'rewarded_modality'
    """ designate label to decode; corresponds to column in the trials table"""
    spike_count_intervals: Literal['pre_stim_single_bin', 'binned_stim_and_response', 'pre_stim_single_bin_0.5', 'pre_stim_single_bin_1.5', 'binned_stim_and_response_0.025', 'binned_stim_and_response_0.5','binned_stim_0.5','binned_stim_0.1','binned_stim_0.05','binned_stim_only_0.05','binned_stim_only_0.025','binned_stim_only_0.02','binned_stim_only_0.01','binned_stim_only_0.005','binned_stim_onset_only_0.01','binned_stim_onset_only_0.005','binned_response_0.025','binned_prestim_0.1'] = 'pre_stim_single_bin'
    baseline_subtraction: bool = False
    """whether to subtract the average baseline context modulation from each unit/trial"""
    n_blocks_expected: int = 6
    """ set number of blocks expected - defaults to 6"""
    use_cumulative_spike_counts: bool = False
    """ toggle using cumulative spike counts from start of first interval for decoding"""
    sliding_window_size: float | None = None
    """ set sliding time window size if different from step size in spike_count_intervals """
    linear_shift: bool = True
    """ toggle linear shift (if False, only runs decoding on aligned trials/ephys) """
    test_on_spontaneous: bool = False
    """ toggle testing the decoder model (trained on the full task) on spontaneous activity """
    scaler: Literal['robust','standard','none'] = 'robust'
    """ set data scaling method: standard = mean/stdev, robust = median/iqr, none = do not scale """


    @property
    def data_path(self) -> upath.UPath:
        """Path to delta lake on S3"""
        return upath.UPath("s3://aind-scratch-data/dynamic-routing/decoding/results") /f"{'_'.join([self.result_prefix, self.run_id])}"

    @property
    def json_path(self) -> upath.UPath:
        """Path to params json on S3"""
        return self.data_path.with_suffix('.json')

    @property
    def min_n_units_query(self) -> Expr:
        if self.unit_subsample_size is None:
            min_ = self.min_n_units
        else:
            #determine number of units required based on max of (min_n_units, unit_subsample_size)
            min_ = max(self.min_n_units,self.unit_subsample_size)
        return pl.col('unit_id').n_unique().over(self.units_group_by).ge(min_)

    @pydantic.computed_field(repr=False)
    @property
    def units_query(self) -> Expr:
        drift_base = (pl.col('decoder_label') != "noise") & (pl.col('isi_violations_ratio') <= 0.5) & (pl.col('amplitude_cutoff') <= 0.1) & (pl.col('presence_ratio') >= 0.7)
        return {
            'medium': (pl.col('isi_violations_ratio') <= 0.5) & (pl.col('presence_ratio') >= 0.9) & (pl.col('amplitude_cutoff') <= 0.1),
            
            'strict': (pl.col('isi_violations_ratio') <= 0.1) & (pl.col('presence_ratio') >= 0.99) & (pl.col('amplitude_cutoff') <= 0.1),
            
            'use_sliding_rp': (pl.col('sliding_rp_violation') <= 0.1) & (pl.col('presence_ratio') >= 0.99) & (pl.col('amplitude_cutoff') <= 0.1),
            
            'recalc_presence_ratio': (pl.col('sliding_rp_violation') <= 0.1) & (pl.col('presence_ratio_task') >= 0.99) & (pl.col('amplitude_cutoff') <= 0.1),
            
            'no_drift': drift_base,
            
            'loose_drift': drift_base & (pl.col('activity_drift') <= 0.2),
            
            'medium_drift': drift_base & (pl.col('activity_drift') <= 0.15),
            
            'strict_drift': drift_base & (pl.col('activity_drift') <= 0.1),
        }[self.unit_criteria]
    
    @pydantic.computed_field(repr=False)
    @property
    def spike_count_interval_configs(self) -> list[BinnedRelativeIntervalConfig]:
        return {
            'pre_stim_single_bin': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.2,
                    stop_time=0,
                    bin_size=0.2,
                ),
            ],
            'pre_stim_single_bin_0.5': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.5,
                    stop_time=0,
                    bin_size=0.5,
                ),
            ],
            'pre_stim_single_bin_1.5': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=0,
                    bin_size=1.5,
                ),
            ],
            'binned_stim_and_response': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.4,
                    stop_time=2.0,
                    bin_size=0.2,
                ),
                BinnedRelativeIntervalConfig(
                    event_column_name='response_time',
                    start_time=-0.4,
                    stop_time=2.0,
                    bin_size=0.2,
                ),
            ],
            'binned_stim_and_response_0.025': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=2.0,
                    bin_size=0.025,
                ),
                BinnedRelativeIntervalConfig(
                    event_column_name='response_or_reward_time',
                    start_time=-1.0,
                    stop_time=1.0,
                    bin_size=0.025,
                ),
            ],
            'binned_stim_and_response_0.5': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=5.5,
                    bin_size=0.5,
                ),
                BinnedRelativeIntervalConfig(
                    event_column_name='response_or_reward_time',
                    start_time=-2.5,
                    stop_time=4.5,
                    bin_size=0.5,
                ),
            ],
            'binned_stim_0.5': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=5.5,
                    bin_size=0.5,
                ),
            ],
            'binned_stim_0.1': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=5.5,
                    bin_size=0.1,
                ),
            ],
            'binned_stim_0.05': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=5.5,
                    bin_size=0.05,
                ),
            ],
            'binned_stim_only_0.05': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=0.6,
                    bin_size=0.05,
                ),
            ],
            'binned_stim_only_0.025': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=0.6,
                    bin_size=0.025,
                ),
            ],
            'binned_stim_only_0.02': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=0.6,
                    bin_size=0.02,
                ),
            ],
            'binned_stim_only_0.01': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=0.6,
                    bin_size=0.01,
                ),
            ],
            'binned_stim_only_0.005': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.1,
                    stop_time=0.6,
                    bin_size=0.005,
                ),
            ],
            'binned_stim_onset_only_0.01': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.05,
                    stop_time=0.3,
                    bin_size=0.01,
                ),
            ],
            'binned_stim_onset_only_0.005': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-0.05,
                    stop_time=0.3,
                    bin_size=0.005,
                ),
            ],
            'binned_response_0.025': [
                BinnedRelativeIntervalConfig(
                    event_column_name='response_or_reward_time',
                    start_time=-1.0,
                    stop_time=1.0,
                    bin_size=0.025,
                ),
            ],
            'binned_prestim_0.1': [
                BinnedRelativeIntervalConfig(
                    event_column_name='stim_start_time',
                    start_time=-1.5,
                    stop_time=0.0,
                    bin_size=0.1,
                ),
            ],
        }[self.spike_count_intervals]

    @pydantic.computed_field(repr=False)
    def datacube_version(self) -> str:
        return utils.get_datacube_dir().name.split('_')[-1]
        
    # set the priority of the sources:
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        *args,
        **kwargs,
    ):
        # the order of the sources is what defines the priority:
        # - first source is highest priority
        # - for each field in the class, the first source that contains a value will be used
        return (
            init_settings,
            pydantic_settings.sources.JsonConfigSettingsSource(settings_cls, json_file='parameters.json'),
            pydantic_settings.CliSettingsSource(settings_cls, cli_parse_args=True),
        )
        
# end of run params ------------------------------------------------ #

def group_structures(frame: polars._typing.FrameType, keep_originals=True) -> polars._typing.FrameType:
    grouping = {
        'SCop': 'SCs',
        'SCsg': 'SCs',
        'SCzo': 'SCs',
        'SCig': 'SCm',
        'SCiw': 'SCm',
        'SCdg': 'SCm',
        'SCdw': 'SCm',
        "ECT1": 'ECT',
        "ECT2/3": 'ECT',    
        "ECT6b": 'ECT',
        "ECT5": 'ECT',
        "ECT6a": 'ECT', 
        "ECT4": 'ECT',
    }
    n_repeats = 2 if keep_originals else 1
    frame = (
        frame
        .with_columns(
            pl.when(pl.col('structure').is_in(grouping))
            .then(pl.col('structure').repeat_by(n_repeats))
            .otherwise(pl.col('structure').repeat_by(1))
        )
        .explode('structure')
        .with_columns(
            pl.when(pl.col('structure').is_in(grouping).is_first_distinct().over('unit_id'))
            .then(pl.col('structure').replace(grouping))
            .otherwise(pl.col('structure'))
        )
    
    )
    return frame 

def repeat_multi_probe_areas(frame: polars._typing.FrameType) -> polars._typing.FrameType:
    """"If an area is recorded on multiple probes, transform the dataframe so it has rows for each
    probe and a row for both probes combined ('electrode_group_names': List[String])"""
    duplicates =  (
        frame
        .clone()
        .with_columns(
             pl.col('electrode_group_name').unique().over('session_id', 'structure', mapping_strategy='join').alias('electrode_group_names')
         )
        .filter(
             pl.col('electrode_group_names').list.len().ge(2)
         )
    )
    return (
        pl.concat(
            [
                frame.with_columns(pl.col('electrode_group_name').cast(pl.List(pl.String)).alias('electrode_group_names')),
                duplicates,
            ],
        )
        .drop('electrode_group_name')
    )

def decode_context_with_linear_shift(
    session_ids: str | Iterable[str],
    params: Params,
) -> None:
    if isinstance(session_ids, str):
        session_ids = [session_ids]

    combinations_df = (
        utils.get_df('units', lazy=True)
        .drop_nulls('structure')
        .filter(
            pl.col('session_id').is_in(session_ids),
            params.units_query,
        )
        .pipe(group_structures, keep_originals=True)
        .pipe(repeat_multi_probe_areas)
        .filter(params.min_n_units_query)
        .select(params.units_group_by)
        .unique(params.units_group_by)
        .collect()
    )
    if params.skip_existing and params.data_path.exists():
        existing = (
            pl.scan_parquet(params.data_path.as_posix().removesuffix('/') + '/')
            .filter(
                pl.col('unit_subsample_size').is_null() if params.unit_subsample_size is None else pl.col('unit_subsample_size').eq(params.unit_subsample_size),
                pl.col('unit_criteria') == params.unit_criteria,
            )
            .select(params.units_group_by)
            .unique(params.units_group_by)
            .collect()
            .to_dicts()
        )
    else:
        existing = []
    def is_row_in_existing(row):
        """Regular dict comparison doesn't work with list field?"""
        return any(
            x for x in existing 
            if all(x[k] == row[k] for k in ['session_id', 'structure'])
            and set(x['electrode_group_names']) == set(row['electrode_group_names'])
        )
        
    logger.info(f"Processing {len(combinations_df)} unique session/area/probe combinations")
    if params.use_process_pool:
        session_results: dict[str, list[cf.Future]] = {}
        future_to_session = {}
        lock = None # multiprocessing.Manager().Lock() # or None
        with cf.ProcessPoolExecutor(max_workers=params.max_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            for row in combinations_df.iter_rows(named=True):
                if params.skip_existing and is_row_in_existing(row):
                    logger.info(f"Skipping {row} - results already exist")
                    continue
                future = executor.submit(
                    wrap_decoder_helper,
                    params=params,
                    **row,
                    lock=lock,
                )
                session_results.setdefault(row['session_id'], []).append(future)
                future_to_session[future] = row['session_id']
                logger.debug(f"Submitted decoding to process pool for session {row['session_id']}, structure {row['structure']}")
                if params.test:
                    logger.info("Test mode: exiting after first session")
                    break
            for future in tqdm.tqdm(cf.as_completed(future_to_session), total=len(future_to_session), unit='structure', desc=f'Decoding'):
                session_id = future_to_session[future]
                if all(future.done() for future in session_results[session_id]):
                    logger.debug(f"Decoding completed for session {session_id}")
                    for f in session_results[session_id]:
                        try:
                            _ = f.result()
                        except Exception:
                            logger.exception(f'{session_id} | Failed:')
                    logger.info(f'{session_id} | Completed')

    else: # single-process mode
        for row in tqdm.tqdm(combinations_df.iter_rows(named=True), total=len(combinations_df), unit='row', desc=f'decoding'):
            if params.skip_existing and is_row_in_existing(row):
                logger.info(f"Skipping {row} - results already exist")
                continue
            try:
                wrap_decoder_helper(
                    params=params,
                    **row,
                )
            except NotEnoughBlocksError as exc:
                logger.warning(f'{row["session_id"]} | {exc!r}')
            except Exception:
                logger.exception(f'{row["session_id"]} | Failed:')
            if params.test:
                logger.info("Test mode: exiting after first session")
                break

def wrap_decoder_helper(
    params: Params,
    session_id: str,
    structure: str,
    electrode_group_names: Sequence[str],
    lock = None,
) -> None:
    logger.debug(f"Getting units and trials for {session_id} {structure}")
    results = []

    all_trials = (
        utils.get_df('trials', lazy=True)
        .filter(
            pl.col('session_id') == session_id,
        ).with_columns( #make new columns for is_response_or_reward and response_or_reward_time
            pl.min_horizontal('response_time','reward_time').alias('response_or_reward_time'),
            (pl.col('response_time').is_not_null() | pl.col('reward_time').is_not_null()).alias('is_response_or_reward')
        ).filter(
            params.trials_filter,
            # obs_intervals may affect number of trials available
        )
        .sort('trial_index')
        .collect()
    )

    if params.test_on_spontaneous:
        #make random seed from the session id for consistency
        random_seed = int(session_id.replace('_','').replace('-',''))
        spont_flag=True
        try:
            spont_trials = pl.from_pandas(data_utils.generate_spontaneous_trials_table(session_id,distribution='DR',random_seed=random_seed))
            interval_bin_size = params.spike_count_interval_configs[0].bin_size
            spike_counts_spont_df = (
                utils.get_per_trial_spike_times(
                    intervals={
                        'n_spikes_window': (
                            pl.col('start_time') + 0, 
                            pl.col('start_time') + interval_bin_size,
                        ),
                    },
                    trials_frame=spont_trials,
                    as_counts=True,
                    unit_ids=(
                        utils.get_df('units', lazy=True)
                        .pipe(group_structures)
                        .filter(
                            params.units_query,
                            pl.col('session_id') == session_id,
                            pl.col('structure') == structure,
                            pl.col('electrode_group_name').is_in(electrode_group_names),
                        )
                        .select('unit_id')
                        .collect()
                        ['unit_id']
                        .unique()
                    ),
                )
                .filter(
                    pl.col('n_spikes_window').is_not_null(),
                    # only keep observed trials
                )
                .sort('trial_index', 'unit_id') 
            )
        except:
            logger.info(f"No spontaneous epoch for {session_id}; skipping session")
            spont_flag=False
            spont_data=None
            return
        
    else:
        spont_flag=False
        spont_data = None



    if params.label_to_decode == 'context_appropriate_for_response':
        all_trials=(
            all_trials.filter(
                pl.col('is_target'),
                pl.col('is_hit').eq(False)
            ).with_columns(
                (((pl.col("is_vis_target")==True) & (pl.col("is_response")==True)) |
                ((pl.col("is_aud_target")==True) & (pl.col("is_response")==False))
                ).alias("is_vis_appropriate_response"),
            ).with_columns(
                (pl.when(pl.col("is_vis_appropriate_response")==True)
                    .then(pl.lit("vis"))
                    .otherwise(pl.lit("aud"))
                ).alias("context_appropriate_for_response")
            )
        )
        #if only one block, this method won't work so cancel for this structure/session
        if all_trials.n_unique('block_index') != 6:
            raise NotEnoughBlocksError(f'Expecting 6 blocks for context_appropriate_for_response analysis: {session_id} has {trials.n_unique("block_index")} blocks of observed ephys data')


    
    # select unit ids for resampling here - keep consistent across time bins
    resample_unit_ids=[]
    unique_unit_ids=(
        utils.get_df('units', lazy=True)
        .pipe(group_structures)
        .filter(
            params.units_query,
            pl.col('session_id') == session_id,
            pl.col('structure') == structure,
            pl.col('electrode_group_name').is_in(electrode_group_names),
        )
        .select('unit_id')
        .sort('unit_id')
        .collect()
        ['unit_id']
        .unique()
    )
    
    n_units_to_use = params.unit_subsample_size or len(unique_unit_ids) # if unit_subsample_size is None, use all available        
    unit_idx = list(range(0, len(unique_unit_ids)))

    for repeat_idx in range(params.n_repeats):
        sel_unit_idx = random.sample(unit_idx, n_units_to_use)
        resample_unit_ids.append(unique_unit_ids[sel_unit_idx])
    resample_unit_ids=np.array(resample_unit_ids)
        

    for interval_config in params.spike_count_interval_configs:
        for start, stop in interval_config.intervals:
            
            #option to use cumulative spike counts
            #change start to equal the start of the first interval
            if params.use_cumulative_spike_counts and params.sliding_window_size is not None:
                 logger.exception(f'cumulative_spike_counts and sliding_window_size are incompatible, select only one to use')

            if params.use_cumulative_spike_counts:
                start_original=np.copy(start)
                start=interval_config.intervals[0][0]
            
            elif params.sliding_window_size is not None:
                start_original=np.copy(start)
                stop_original=np.copy(stop)
                start=(start_original+stop_original)/2-(params.sliding_window_size/2)
                stop=(start_original+stop_original)/2+(params.sliding_window_size/2)
                #start=stop-params.sliding_window_size
                

            #option to subtract trialwise baseline, defined as 500ms before event (stimulus)
            if params.baseline_subtraction:
                spike_counts_df = (
                    utils.get_per_trial_spike_times(
                        intervals={
                            'n_spikes_baseline': (
                                pl.col(interval_config.event_column_name) + -0.5, 
                                pl.col(interval_config.event_column_name) + 0
                            ),
                            'n_spikes_window': (
                                pl.col(interval_config.event_column_name) + start, 
                                pl.col(interval_config.event_column_name) + stop,
                            ),
                        },
                        trials_frame=all_trials,
                        as_counts=True,
                        unit_ids=(
                            utils.get_df('units', lazy=True)
                            .pipe(group_structures)
                            .filter(
                                params.units_query,
                                pl.col('session_id') == session_id,
                                pl.col('structure') == structure,
                                pl.col('electrode_group_name').is_in(electrode_group_names),
                            )
                            .select('unit_id')
                            .collect()
                            ['unit_id']
                            .unique()
                        ),
                    )
                    .with_columns(
                        pl.col('n_spikes_baseline')*2*(stop-start),
                        #get firing rate, then extrapolate spike counts to the bin size used
                    )
                    .with_columns(
                        pl.col('n_spikes_window').sub(pl.col('n_spikes_baseline'))
                    )
                    .filter(
                        pl.col('n_spikes_window').is_not_null(),
                        # only keep observed trials
                    )
                    .sort('trial_index', 'unit_id') 
                )

            else:

                spike_counts_df = (
                    utils.get_per_trial_spike_times(
                        intervals={
                            'n_spikes_window': (
                                pl.col(interval_config.event_column_name) + start, 
                                pl.col(interval_config.event_column_name) + stop,
                            ),
                        },
                        trials_frame=all_trials,
                        as_counts=True,
                        unit_ids=(
                            utils.get_df('units', lazy=True)
                            .pipe(group_structures)
                            .filter(
                                params.units_query,
                                pl.col('session_id') == session_id,
                                pl.col('structure') == structure,
                                pl.col('electrode_group_name').is_in(electrode_group_names),
                            )
                            .select('unit_id')
                            .collect()
                            ['unit_id']
                            .unique()
                        ),
                    )
                    .filter(
                        pl.col('n_spikes_window').is_not_null(),
                        # only keep observed trials
                    )
                    .sort('trial_index', 'unit_id') 
                )
                # len == n_units x n_trials, with spike counts in a column
                # sequence of unit_ids is used later: don't re-sort!

            
            logger.debug(f"Got spike counts: {spike_counts_df.shape} rows")

            trials = (
                all_trials
                .filter(
                    pl.col('session_id') == session_id,
                    pl.col('trial_index').is_in(spike_counts_df['trial_index'].unique()),
                    # obs_intervals may affect number of trials available
                )
                .sort('trial_index')
                .select(params.label_to_decode, 'start_time', 'trial_index', 'block_index', 'session_id')
            )

            if (
                trials['block_index'].n_unique() == 1
                and not (
                    utils.get_df('session')
                    .filter(
                        pl.col('session_id') == trials['session_id'][0],
                        pl.col('keywords').list.contains('templeton'),
                    )
                ).is_empty()
            ):
                logger.info(f'Adding dummy context labels for Templeton session {session_id}')
                trials = (
                    trials
                    .with_columns(
                        pl.col('start_time').sub(pl.col('start_time').min().over('session_id')).truediv(10*60).floor().clip(0, 5).alias('block_index')
                        # short 7th block will sometimes be present: merge into 6th with clip
                    )
                    .with_columns(
                        pl.when(pl.col('block_index').mod(2).eq(random.choice([0, 1])))
                        .then(pl.lit('vis'))
                        .otherwise(pl.lit('aud'))
                        .alias('rewarded_modality')
                    )
                    .sort('trial_index')
                )
            if trials.n_unique('block_index') != params.n_blocks_expected:
                raise NotEnoughBlocksError(f'Expecting {params.n_blocks_expected} blocks: {session_id} {structure} has {trials.n_unique("block_index")} blocks of observed ephys data')
            logger.debug(f"Got {len(trials)} trials")

            label_to_decode = trials[params.label_to_decode].to_numpy().squeeze()

            if params.linear_shift:
                max_neg_shift = math.ceil(len(trials.filter(pl.col('block_index')==0))/2)
                max_pos_shift = math.floor(len(trials.filter(pl.col('block_index')==5))/2)
            else:
                max_neg_shift = 0
                max_pos_shift = 1
            shifts = tuple(range(-max_neg_shift, max_pos_shift + 1))
            logger.debug(f"Using shifts from {shifts[0]} to {shifts[-1]}")

            for repeat_idx in tqdm.tqdm(range(params.n_repeats), total=params.n_repeats, unit='repeat', desc=f'repeating {structure} | {session_id}'):
            
                filtered_unit_df = spike_counts_df.filter(pl.col('unit_id').is_in(resample_unit_ids[repeat_idx]))

                spike_counts_array = (
                    filtered_unit_df
                    .select('n_spikes_window')
                    .to_numpy()
                    .squeeze()
                    .reshape(filtered_unit_df.n_unique('trial_index'), filtered_unit_df.n_unique('unit_id'))
                )
                logger.debug(f"Reshaped spike counts array: {spike_counts_array.shape}")
                
                unit_ids = filtered_unit_df['unit_id'].unique(maintain_order=True).to_list()

                logger.debug(f"Repeat {repeat_idx}: selected {len(sel_unit_idx)} units")

                if spont_flag:
                    filtered_spont_unit_df=spike_counts_spont_df.filter(pl.col('unit_id').is_in(resample_unit_ids[repeat_idx]))
                    spont_data = (
                        filtered_spont_unit_df
                        .select('n_spikes_window')
                        .to_numpy()
                        .squeeze()
                        .reshape(filtered_spont_unit_df.n_unique('trial_index'), filtered_spont_unit_df.n_unique('unit_id'))
                    )
                
                for shift in (*shifts, None): # None will be a special case using all trials, with no shift
                    
                    is_all_trials = shift is None
                    if not is_all_trials:
                        #if not params.linear_shift:
                            #continue
                        labels = label_to_decode[max_neg_shift: -max_pos_shift]
                        if params.crossval=='blockwise':
                            crossval_index=trials['block_index'].to_numpy().squeeze()[max_neg_shift: -max_pos_shift]
                        else:
                            crossval_index=None
                        first_trial_index = max_neg_shift + shift
                        last_trial_index = len(trials) - max_pos_shift + shift
                        logger.debug(f"Shift {shift}: using trials {first_trial_index} to {last_trial_index} out of {len(trials)}")
                        assert first_trial_index >= 0, f"{first_trial_index=}"
                        assert last_trial_index > first_trial_index, f"{last_trial_index=}, {first_trial_index=}"
                        assert last_trial_index <= spike_counts_array.shape[0], f"{last_trial_index=}, {spike_counts_array.shape[0]=}"
                        data = spike_counts_array[first_trial_index: last_trial_index, :]
                    else:
                        labels = label_to_decode
                        crossval_index=trials['block_index'].to_numpy().squeeze()
                        data = spike_counts_array[:, :]

                    assert data.shape == (len(labels), len(unit_ids)), f"{data.shape=}, {len(labels)=}, {len(sel_unit_idx)=}"
                    logger.debug(f"Shift {shift}: using data shape {data.shape} with {len(labels)} labels")

                    _result = decoder_helper(
                        data,
                        labels,
                        decoder_type=params.decoder_type,
                        crossval=params.crossval,
                        crossval_index=crossval_index,
                        labels_as_index=params.labels_as_index,
                        train_test_split_input=None,
                        regularization=params.regularization,
                        penalty=params.penalty,
                        solver=params.solver,
                        n_jobs=None,
                        other_data=spont_data,
                        scaler=params.scaler,
                    )
                    result = {}
                    result['balanced_accuracy_test'] = _result['balanced_accuracy_test'].item()
                    result['balanced_accuracy_train'] = _result['balanced_accuracy_train'].item()
                    result['time_aligned_to'] = interval_config.event_column_name
                    result['bin_size'] = interval_config.bin_size
                    result['sliding_window_size'] = params.sliding_window_size
                    if params.use_cumulative_spike_counts:
                        result['bin_center'] = stop
                    elif params.sliding_window_size is not None:
                        result['bin_center'] = (start_original + stop_original) / 2
                        #result['bin_center'] = stop
                    else:
                        result['bin_center'] = (start + stop) / 2
                    result['shift_idx'] = shift
                    result['repeat_idx'] = repeat_idx
                    result['labels'] = _result['labels'].tolist()
                    
                    if shift in (0, None):
                        if params.label_to_decode in ["is_response", "is_target", "is_rewarded"]:
                            result['decision_function'] = _result['decision_function'].tolist()
                            result['decision_function_all'] = _result['decision_function_all'].tolist()
                            result['predict_proba'] = _result['predict_proba'][:, np.where(_result['label_names'] == True)[0][0]].tolist()
                            result['predict_proba_all_trials'] = _result['predict_proba_all_trials'][:, np.where(_result['label_names'] == True)[0][0]].tolist()
                        elif params.label_to_decode=="stim_name":
                            #if decoding only 2 stimuli
                            if len(_result['label_names'])==2: 
                                result['decision_function'] = _result['decision_function'].tolist()
                                result['decision_function_all'] = _result['decision_function_all'].tolist()
                                if 'vis1' in _result['label_names']:
                                    temp_target_label='vis1'
                                elif 'sound1' in _result['label_names']:
                                    temp_target_label='sound1'
                                result['predict_proba'] = _result['predict_proba'][:, np.where(_result['label_names'] == temp_target_label)[0][0]].tolist()
                                result['predict_proba_all_trials'] = _result['predict_proba_all_trials'][:, np.where(_result['label_names'] == temp_target_label)[0][0]].tolist()
                            #if decoding all 4 stimuli
                            elif len(_result['label_names'])==4:
                                predict_proba_multiclass=np.full((len(labels),4),np.nan)
                                predict_proba_all_trials_multiclass=np.full((len(labels),4),np.nan)
                                stim_order=['sound1','sound2','vis1','vis2']
                                for ss,stim_label in enumerate(stim_order):
                                    result['decision_function_'+stim_label] = _result['decision_function'][:, np.where(_result['label_names'] == stim_label)[0][0]].tolist()
                                    result['decision_function_all_'+stim_label] = _result['decision_function_all'][:, np.where(_result['label_names'] == stim_label)[0][0]].tolist()
                                    result['predict_proba_'+stim_label] = _result['predict_proba'][:, np.where(_result['label_names'] == stim_label)[0][0]].tolist()
                                    result['predict_proba_all_trials_'+stim_label] = _result['predict_proba_all_trials'][:, np.where(_result['label_names'] == stim_label)[0][0]].tolist()

                        elif params.label_to_decode in ["rewarded_modality","context_appropriate_for_response"]:
                            result['decision_function'] = _result['decision_function'].tolist()
                            result['decision_function_all'] = _result['decision_function_all'].tolist()
                            result['predict_proba'] = _result['predict_proba'][:, np.where(_result['label_names'] == 'vis')[0][0]].tolist()
                            result['predict_proba_all_trials'] = _result['predict_proba_all_trials'][:, np.where(_result['label_names'] == 'vis')[0][0]].tolist()
                        else:
                            logger.exception(f'{session_id} | Failed: decoding unknown column')

                        if params.test_on_spontaneous: #only save spontaneous results columns if they were computed
                            if spont_data is not None: #save predictions about spont data plus relevant trial info
                                result['predict_proba_spont'] = _result['predict_proba_other'][:, np.where(_result['label_names'] == 'vis')[0][0]].tolist()
                                result['decision_function_spont'] = _result['decision_function_other'].tolist()
                                result['pred_label_spont'] = _result['label_names'][_result['pred_label_other']].tolist()
                                result['spont_trial_times'] = spont_trials['start_time'].to_list()
                                result['spont_epoch_name'] = spont_trials['epoch_name'].to_list()
                                result['spont_trial_is_rewarded'] = spont_trials['is_rewarded'].to_list()
                            else: 
                                result['predict_proba_spont'] = []
                                result['decision_function_spont'] = []
                                result['pred_label_spont'] = []
                                result['spont_trial_times'] = []
                                result['spont_epoch_name'] = []
                                result['spont_trial_is_rewarded'] = []
                    else:
                        # don't save probabilities from shifts which we won't use 
                        result['predict_proba'] = None 
                        result['predict_proba_all_trials'] = None
                        result['decision_function'] = None
                        result['decision_function_all'] = None 
                        
                    if is_all_trials:
                        result['trial_indices'] = trials['trial_index'].to_list()
                    elif shift in (0, None):
                        result['trial_indices'] = trials['trial_index'].to_list()[first_trial_index: last_trial_index]
                    else:
                        # don't save trial indices for all shifts
                        result['trial_indices'] = None 
                        
                    result['unit_ids'] = unit_ids
                    result['coefs'] = _result['coefs'][0].tolist()
                    result['is_all_trials'] = is_all_trials
                    results.append(result)
                    if params.test:
                        break
                if params.test:
                    break
            if params.test:
                logger.info(f"Test mode: exiting after first bin in relative to {interval_config.event_column_name}")
                break
        if params.test:
            logger.info("Test mode: exiting after first event intervals config")
            break
        
    with lock or contextlib.nullcontext():
        logger.info('Writing data')
        (
            pl.DataFrame(results)
            .with_columns(
                pl.lit(session_id).alias('session_id'),
                pl.lit(structure).alias('structure'),
                pl.lit(sorted(electrode_group_names)).alias('electrode_group_names'),
                pl.lit(params.unit_subsample_size).alias('unit_subsample_size').cast(pl.UInt16),
                pl.lit(params.unit_criteria).alias('unit_criteria'),
            )
            .cast(
                {
                    'shift_idx': pl.Int16,
                    'repeat_idx': pl.UInt16,
                    'time_aligned_to': pl.Enum([c.event_column_name for c in params.spike_count_interval_configs]),
                    'trial_indices': pl.List(pl.UInt16),
                    'predict_proba': pl.List(pl.Float64),
                    'coefs': pl.List(pl.Float64),
                }
            )
            .write_parquet(
                (params.data_path / f"{uuid.uuid4()}.parquet").as_posix(),
                compression_level=18,
                statistics='full',    
            )
            # .write_delta(params.data_path.as_posix(), mode='append')
        )
    logger.info(f"Completed decoding for session {session_id}, structure {structure}")
    # return results