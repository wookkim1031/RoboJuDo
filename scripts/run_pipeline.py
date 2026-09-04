# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1",
        help="Name of the config class to use",
    )
    parser.add_argument(
        "-env",
        type=str,
        default=None,
        help="Netzwerk-Interface zum Roboter (z.B. enp5s0)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)

    cfg: RlPipelineCfg = config_manager.get_cfg()

    pipeline_type = cfg.pipeline_type

    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    if not cfg.env.is_sim and args.env is not None:
        cfg.env.unitree.net_if = args.env
        logger.info(f"net_if override: {args.env}")

    pipeline = pipeline_class(cfg=cfg)

    if not cfg.env.is_sim:
        pipeline.prepare()
    elif getattr(pipeline, "_has_default_pose_mode", False):
        pipeline._set_default_pose_mode(True)
        logger.warning("Sim mode — holding default pose, press R to start motion")

    try:
        while True:
            time_start = time.time()
            pipeline.step()
            time_end = time.time()
            time_diff = time_end - time_start

            # keep the pipeline running at the desired frequency
            if not cfg.run_fullspeed:
                time_diff = pipeline.dt - time_diff
                if time_diff > 0:
                    time.sleep(time_diff)
                else:
                    if not cfg.env.is_sim:
                        logger.error(f"Warning: frame drop -> {time_diff}")
                        if time_diff < -0.2:
                            logger.critical("Exiting due to excessive frame drop")
                            pipeline.env.shutdown()
                            time.sleep(10)
                            break

    except KeyboardInterrupt:
        logger.warning("Keyboard interupt")

    except Exception:
        logger.exception("error in for loop")

    finally:
        pipeline.env.shutdown()


if __name__ == "__main__":
    main()
