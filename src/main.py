from lightning.pytorch.cli import LightningCLI

if __name__ == '__main__':
    cli = LightningCLI(save_config_kwargs={"save_to_log_dir": True, "overwrite": True})