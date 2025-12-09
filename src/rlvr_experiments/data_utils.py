# temporary
def build_titan_dataloader(trainer):
    job_config = trainer.job_config
    train_spec = trainer.train_spec 

    # figure out dp_world_size / dp_rank exactly like original Trainer
    parallel_dims = trainer.parallel_dims
    world_mesh = parallel_dims.world_mesh

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_world_size = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_world_size, dp_rank = 1, 0

    dataloader = train_spec.build_dataloader_fn(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=trainer.tokenizer,
        job_config=job_config,
    )
    return dataloader