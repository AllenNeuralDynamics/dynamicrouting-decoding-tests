from __future__ import annotations

import os
os.environ['RUST_BACKTRACE'] = '1'
#os.environ['POLARS_MAX_THREADS'] = '1'
os.environ['TOKIO_WORKER_THREADS'] = '1' 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['RAYON_NUM_THREADS'] = '1'


# stdlib imports --------------------------------------------------- #
import dataclasses
import datetime
import json
import logging
import pathlib
import time
from typing import Annotated, Literal

# 3rd-party imports necessary for processing ----------------------- #
import matplotlib
import pandas as pd
import polars as pl
import pydantic_settings
import pydantic_settings.sources
import pydantic.functional_serializers
import upath

# local modules ---------------------------------------------------- #
import utils
import decoding_utils


# logging configuration -------------------------------------------- #
# use `logger.info(msg)` instead of `print(msg)` so we get timestamps and origin of log messages
logger = logging.getLogger(
    pathlib.Path(__file__).stem if __name__.endswith("_main__") else __name__
    # multiprocessing gives name '__mp_main__'
)

# general configuration -------------------------------------------- #
matplotlib.rcParams['pdf.fonttype'] = 42
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR) # suppress matplotlib font warnings on linux


    
# define run params here ------------------------------------------- #
Expr = Annotated[
    pl.Expr, pydantic.functional_serializers.PlainSerializer(lambda expr: expr.meta.serialize(format='json'), return_type=str)
]

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
    max_workers: int | None = pydantic.Field(None, exclude=True, repr=True)
    """For process pool"""

    # Decoding parameters ----------------------------------------------- #
    session_table_query: str = "is_ephys & is_task & is_annotated & is_production & issues=='[]'"
    unit_criteria: str = pydantic.Field("loose_drift", exclude=True, repr=True) # often varied, stored in data not params file
    unit_subsample_size: int | None = pydantic.Field(None, exclude=True, repr=True) # often varied, so will be stored with data, not in the params file
    """number of units to sample for each area"""
    n_repeats: int = 25
    """number of times to repeat decoding with different randomly sampled units"""
    min_n_units: int = 5 
    """only process areas with at least this many units"""
    input_data_type: Literal['spikes', 'facemap', 'LP'] = 'spikes'
    spikes_time_before: float = 0.2
    crossval: Literal['5_fold', 'blockwise'] = '5_fold'
    """blockwise untested with linear shift"""
    labels_as_index: bool = True
    """convert labels (context names) to index [0,1]"""
    decoder_type: Literal['linearSVC', 'LDA', 'RandomForest', 'LogisticRegression'] = 'LogisticRegression'
    regularization: float | None = None
    """ set regularization (C) for the decoder. Setting to None reverts to the default value (usually 1.0) """
    penalty: str | None = None
    """ set penalty for the decoder. Setting to None reverts to default """
    solver: str | None = None
    """ set solver for the decoder. Setting to None reverts to default """
    units_group_by: list[str] = ['session_id', 'structure', 'electrode_group_names']
    
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
            min_ = self.min_n_units + self.unit_subsample_size
        return pl.col('unit_id').n_unique().over(self.units_group_by).ge(min_)

    @property
    def units_query(self) -> Expr:
        return self.unit_criteria_queries[self.unit_criteria]

    @pydantic.computed_field(repr=False)
    @property
    def unit_criteria_queries(self) -> dict[str, Expr]:
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
        }
    
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
        
        
# processing function ---------------------------------------------- #

def main():
    t0 = time.time()
    
    utils.setup_logging()
    params = Params() # reads from CLI args
    logger.setLevel(params.logging_level)
    
    if params.override_params_json:
        logger.info(f"Overriding parameters with {params.override_params_json}")
        params = Params(**json.loads(params.override_params_json))
        
    if params.test:
        params = Params(
            result_prefix=f"test/{params.result_prefix}",
            min_n_units=20,
            n_repeats=1,
        )
        logger.info("Test mode: using modified set of parameters")
        
    
    # if session_id is passed as a command line argument, we will only process that session,
    # otherwise we process all session IDs that match filtering criteria:    
    session_table = pd.read_parquet(utils.get_datacube_dir() / 'session_table.parquet')
    session_table['issues']=session_table['issues'].astype(str)
    session_ids: list[str] = session_table.query(params.session_table_query)['session_id'].values.tolist()
    logger.debug(f"Found {len(session_ids)} session_ids available for use after filtering")
    
    if params.session_id is not None:
        if params.session_id not in session_ids:
            logger.warning(f"{params.session_id!r} not in filtered session_ids: exiting")
            exit()
        logger.info(f"Using single session_id {params.session_id} provided via command line argument")
        session_ids = [params.session_id]
    elif utils.is_pipeline(): 
        # only one nwb will be available 
        session_ids = set(session_ids) & set(p.stem for p in utils.get_nwb_paths())
    else:
        logger.info(f"Using list of {len(session_ids)} session_ids after filtering")
    
    upath.UPath('/results/params.json').write_text(params.model_dump_json(indent=4))
    if params.json_path.exists():
        existing_params = json.loads(params.json_path.read_text())
        if existing_params != params.model_dump():
            raise ValueError(f"Params file already exists and does not match current params:\n{existing_params=}\n{params.model_dump()=}")
    else:            
        logger.info(f'Writing params file: {params.json_path}')
        params.json_path.write_text(params.model_dump_json(indent=4))
    
    logger.info(f'starting decode_context_with_linear_shift with {params!r}')
    decoding_utils.decode_context_with_linear_shift(session_ids=session_ids, params=params)
    
    utils.ensure_nonempty_results_dir()
    logger.info(f"Time elapsed: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
