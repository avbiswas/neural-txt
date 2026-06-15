from neuraltxt import NeuralTxtReward, NeuralTxt


# passage = """
# We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
# """

response = """Precision is 50% of this particular BERT model."""

reference ="""This BERT is around 60% accurate"""


 
def main() -> None:
    # model = NeuralTxt()
    rm = NeuralTxtReward()
    # answers = model.rephrase(
    #     passage=passage,
    #     temperature=0.7,
    #     rollouts=4
    # )
    rewards = rm.score(response, reference)
    print(rewards)

        


if __name__ == "__main__":
    main()
