from rl_mdp.util import create_mdp
from rl_mdp.util import create_policy_1
from rl_mdp.util import create_policy_2
from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator

def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    mdp = create_mdp()
    policy1 = create_policy_1()
    policy2 = create_policy_2()

    #7a
    # evaluator = MCEvaluator(mdp)

    #7b
    evaluator = TDEvaluator(mdp, alpha=0.1)

    n_episodes = 10
    evaluation1 = evaluator.evaluate(policy1, n_episodes)
    evaluation2 = evaluator.evaluate(policy2, n_episodes)
    print(evaluation1)
    print(evaluation2)


if __name__ == "__main__":
    main()
