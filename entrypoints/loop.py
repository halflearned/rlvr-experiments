def allocate():
    # initialize policy, reference, episode buffer, etc
    reference = Inference(model_path)  # <--- running somewhere
    inference = Inference(model_path) # <--- running somewhere else
    trainer = Trainer(model_path)  # <--- running
    dataset = PromptDataset(dataset_path)


def grpo_loop():

    for iteration in range(num_iterations):

        # weight sync trainer -> reference
        sync_weight(trainer, reference)

        for step in range(num_steps):

            # weight sync trainer -> inference
            sync_weight(trainer, inference)

            # sample a batch
            samples = dataset.sample()

            # get model responses
            output = inference.rollout(samples.prompts)

            # compute rewards
            rewards = verifier.verify(samples, output.answers)

            # perform updates
            for grpo_iteration in range(num_grpo_iterations):

                trainer.train_step(output, rewards)

    
