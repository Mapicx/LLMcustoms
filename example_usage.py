from .llmcustoms.First_Wrapper import FineTuner

tuner = FineTuner(
    data_path="./my_documents/",
    model="auto",
    preset="auto"
)

model_path = tuner.train()