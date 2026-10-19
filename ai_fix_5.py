import os
import sys
import subprocess

def claim_task(task_number):
    with open('issue.txt', 'a') as file:
        file.write(f'claim {task_number}\n')

def run_bounty_script(task_number):
    subprocess.run(['python', 'scripts/bounty.py', 'start', str(task_number)])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <task_number>")
        sys.exit(1)

    task_number = sys.argv[1]
    claim_task(task_number)
    run_bounty_script(task_number)