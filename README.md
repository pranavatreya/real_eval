# Evaluation README

Welcome to the DROID A/B Evaluation System! This README explains how to run the **evaluation client** script, how to configure it for your institution’s setup, and how to provide **high-quality, detailed feedback** that will help us improve our robot learning benchmark.

---

## Installation

1. **Clone or copy** this repository onto your local machine or robot control computer.
2. Ensure you have **Python 3.7+** installed.
3. Install dependencies. Typically you can do:

   ```bash
   pip install -r requirements.txt
   ```

   or simply

   ```bash
   pip install numpy requests Pillow tqdm moviepy matplotlib
   ```

   (If you have a separate environment for robot control, just make sure these libraries are available in that environment.)

4. Make sure **DROID** is installed on your system. This system depends on the `RobotEnv` (or `droid.robot_env`) interface.

---

## Running the Evaluation

1. Create or edit a YAML config file (similar to `berkeley.yaml`) that contains:
   - `logging_server_ip`: **Always** set this to `34.55.101.123:5000`.
   - `third_person_camera`: The default vantage (e.g. `right_image` or `left_image`).
   - A `cameras` section that identifies the camera `name` and `id` for your institution’s camera setup.

   A minimal example, `my_institution.yaml`, might look like:
   ```yaml
   logging_server_ip: 34.55.101.123:5000
   third_person_camera: right_image
   cameras:
     - name: left
       id: 24259877
     - name: right
       id: 24514023
     - name: wrist
       id: 13062452
   ```
   Adjust the IDs to match **your** cameras. Change `third_person_camera` to `left_image` if you prefer the left camera as your default third-person vantage.

2. **Run** the evaluation client script:

   ```bash
   python evaluation_client/main.py my_institution.yaml
   ```

3. **Follow the prompts** in the terminal:
   - Enter your **evaluator name** and **institution** when asked.
   - Confirm that the left/right cameras are correctly pointing at the part of the scene you want for the third-person view.
   - (Optional) **Switch** between the left or right vantage if you prefer to do so; the script will ask you.
   - **Enter** the **language command** you want the policy to follow (e.g., “Pick up the red block and place it in the box.”).
   - The system will then run the A/B evaluation (plus additional policies C and D):
     1. **Policy A** rollout
     2. **Policy B** rollout (then it will ask which policy you preferred, A, B, or tie)
     3. **Policy C** rollout
     4. **Policy D** rollout
   - The script will guide you to provide **binary success** and **partial success** scores. Please keep reading below for partial success guidelines.
   - **At the end**, the script asks whether everything went well and if the session should be considered valid. If you choose “yes,” it is marked as valid; if you choose “no,” it’s marked as invalid.

4. **Repeat** as many times as you want. **Between each run** of the entire script:
   - Feel free to **move the robot** to a new location or **change tasks** for the next A/B evaluation. This fosters diverse evaluations. 
   - You can also reposition or switch cameras to create new viewpoints.

---

## Partial Success Guidelines

When you’re prompted to enter a partial success score in **[0..100]**, here are some guidelines:

- **Pick and Place Example:**
  - 0% if the policy barely moved or obviously failed from the start.
  - 30% if the policy reached for the object but didn’t grasp successfully.
  - 60% if it grasped the object and lifted it but dropped it prematurely or failed to place it accurately.
  - 80–90% if it placed the object very near the goal but slightly missed.
  - 100% only if it did exactly what was asked with no major issues.

These scores don’t have to be exact. They’re a quick measure of **how far** the policy got in the task. Use your best judgement.

---

## Why Are We Evaluating Policies C and D Too?

Although the script is performing an **A/B evaluation** — letting you directly compare policy A vs. policy B (and soliciting your preference) — we also run policies **C** and **D**. We do this so we can:

- **Benchmark** how well our new A/B evaluation approach compares to more traditional robot evaluation methods, where multiple policies are simply run in a fixed scenario.
- Gather more data about overall performance.

In essence, **A/B** is the main approach for user-facing comparisons. **C and D** are there to cross-check and confirm that our benchmark results align with other typical evaluation strategies.

---

## Importance of Long-Form Feedback

After you finish evaluating policies A and B, the script prompts for **long-form textual feedback**. This is critical:

1. **Reference Policy A or Policy B** by name. e.g.:
   - “Policy A followed language instructions more precisely but struggled with dexterous manipulation.”
   - “Policy B seemed robust to odd lighting conditions but sometimes misunderstood the color references.”

2. Provide **granular details**:
   - Did one policy complete the task faster/smoother?
   - Could either policy handle highly out-of-distribution instructions or scenes?
   - Did the policy show good generalization (handling unfamiliar objects or lighting changes)?

This long-form feedback is **extremely** important for us to understand the strengths/weaknesses of each policy beyond raw success rates.

---

## Behind the Scenes

- **All data** (videos, partial success ratings, your textual feedback, etc.) is **automatically logged** to the central server at `34.55.101.123:5000`. You do not need to upload anything manually.
- Each evaluation session times out after a while if incomplete, but typically you’ll end the session yourself.

---

## Final Notes

- Be **impartial**. Avoid any bias based on prior knowledge of who developed which policy. Focus on observed performance.
- We encourage you to:
  - **Change** camera viewpoints, lighting, tasks, or object arrangements **between runs** to get a broad evaluation dataset.
  - **Provide** as much detail as possible in your feedback to help us improve the benchmark.
- If you have **feedback** about how the overall evaluation experience could be improved, or if you run into any problems, please reach out to **Pranav Atreya**.

**Thank you** for your contributions to this benchmark!

---

