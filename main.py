from q_trainer import q_trainer

if __name__ == "__main__":
    trainer = q_trainer(
        state_data_filename='q0.pkl',
        PARAM_number_of_episodes=5,
    )
    trainer.run_session()
    pass