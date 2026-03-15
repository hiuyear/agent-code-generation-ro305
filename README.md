# part of my application to ROP: ro305 project
Automated Agent Generation for Scientific Software
A coding agent receives a plain-English scientific task, selects from a pool of candidate code snippets, executes each one, validates the output, and retries on failure.
The experiment compares two conditions: unguided (no hints) vs guided (one-line domain hint per task).
No external API required. Just numpy, scipy, and matplotlib.
The Core Loop
1. Receive plain-English task description
         +  (optional) one-line domain hint
              |
              v
2. Select a candidate code snippet
   Unguided: random order
   Guided:   hint steers agent to correct candidate first
              |
              v
3. Execute code in isolated namespace
              |
         Valid? --> Done
              |
           Failed
              |
              v
4. Log the failure, try next candidate
This is a simplified prototype of the pipeline in LLM-based agent generation: generate → execute → validate → retry with feedback.
Four Scientific Tasks
Each task has 3 candidate snippets: one correct, two with realistic mistakes.
TaskCorrect approachCommon mistakesMonte Carlo πSample in [0,1], multiply by 4Wrong range [-1,1], missing ×4Gaussian meannp.random.normal(loc=5) + np.mean()Wrong loc parameter, using mediansin(x) integralquad(np.sin, 0, np.pi) unpackedWrong limits, forgetting to unpack tupleEigenvaluesnp.linalg.eigh() for symmetric matrixUsing eig(), wrong matrix
Results
Averaged across 300 runs (unguided order is randomized each run):
MetricUnguidedGuidedFirst-attempt success50.1%100.0%Avg attempts to solve1.681.00
Human guidance cut average attempts to solve in half and eliminated all first-attempt failures.
Show Image
How to Run
bashpip install numpy scipy matplotlib
python agent_codegen.py
Code Structure

TASKS — four scientific tasks, each with candidates and a validator
execute_code() — runs a code string in an isolated namespace, captures errors
validate_*() — task-specific correctness checks with numeric tolerances
CodingAgent — selects candidates, executes, retries on failure
run_experiment() — runs both agents 300 times, tracks success rates

Connection to the Research
This prototype studies the same question as automated agent generation research: does human-provided domain knowledge improve the reliability of agent-generated scientific code?
Here the tasks are small numpy/scipy computations. In a real research setting the agent would be generating code for a library like PySCF, and the domain hints would come from a chemist rather than being pre-written. The pipeline — candidate generation, execution, validation, retry — is identical.
To extend this with a real LLM, replace the candidate pool with live API calls:
python# Instead of shuffling pre-written candidates,
# call an LLM with the task description (and optionally the guidance hint)
# parse the generated code from the response
# feed execution errors back into the next prompt
Everything else — execution, validation, retry logic — stays the same.