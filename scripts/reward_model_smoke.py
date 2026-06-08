from neuraltxt import NeuralTxtReward, NeuralTxt


passage = """
Claude Code, plugged into your services, APIs, and dependencies via MCP. AI Architect, the context layer for coding agents. Production-ready code in 1-shot.
"""

question = "What can Claude Code do"
reference = "Claude Code, plugged into your services, APIs, and dependencies via MCP. AI Architect, the context layer for coding agents. Production-ready code in 1-shot"

def main() -> None:
    model = NeuralTxt()
    rm = NeuralTxtReward()
    answers = model.rephrase(
        passage=passage,
        temperature=0.7,
        rollouts=4
    )
    rewards = rm.rank(answers, passage)
    for r in rewards:
        print()
        print(r.response, r.score)

        


if __name__ == "__main__":
    main()
