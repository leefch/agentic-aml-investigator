"""
Run one case from the terminal, e.g.

    python run_cli.py TXN000123

With no argument it grabs the first thing in the review queue. Handy for a quick
smoke test without spinning up the UI.
"""
import sys

from agent import tools
from agent.graph import investigate


def main():
    if len(sys.argv) > 1:
        txn_id = sys.argv[1]
    else:
        queue = tools.sample_queue(1)
        if not queue:
            print("review queue is empty - generate data first")
            return
        txn_id = queue[0]

    result = investigate(txn_id)

    print(f"\n=== case {txn_id} ===")
    print(f"tier: {result['risk_tier']}  score: {result['risk_score']}")
    print(f"recommendation: {result['recommendation']}  needs_human: {result['needs_human']}")
    print("\nnarrative:\n" + result.get("narrative", ""))
    print("\naudit trail:")
    for step in result["audit"]:
        print(f"  [{step['actor']}] {step['action']} - {step['detail']}")


if __name__ == "__main__":
    main()
