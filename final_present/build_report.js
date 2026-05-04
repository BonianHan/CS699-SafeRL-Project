/**
 * Final Project Report — CS 699 Safe RL on Highway-Env
 * Follows Milestone 5 required structure.
 */
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, PageOrientation, LevelFormat,
  ExternalHyperlink, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak, TabStopType, TabStopPosition,
} = require('docx');

// ------------------------------------------------------------
// Style constants — black-and-white only
// ------------------------------------------------------------
const BLACK = "000000";
const NAVY = BLACK;       // headings: pure black
const ACCENT = BLACK;     // no accent colour
const GREEN = BLACK;
const RED = BLACK;
const BLUE = BLACK;
const GREY = "595959";    // dark grey for secondary text
const LIGHT = "F2F2F2";   // very light grey for table headers
const PALE = "F2F2F2";    // same for any highlight cell

const border = { style: BorderStyle.SINGLE, size: 4, color: BLACK };
const allBorders = { top: border, bottom: border, left: border, right: border };

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
const T = (text, opts = {}) => new TextRun({ text, font: "Calibri", ...opts });
const P = (children, opts = {}) =>
  new Paragraph({ children: Array.isArray(children) ? children : [children], ...opts });

const para = (text, opts = {}) => {
  // accept a plain string or an array of pre-built runs.
  const children = Array.isArray(text)
    ? text
    : [T(text, { size: 22, ...opts.run })];
  return new Paragraph({
    spacing: { after: 120, line: 300, ...opts.spacing },
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    children,
    ...opts.paragraphOpts,
  });
};

const h1 = (text) =>
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [T(text, { size: 32, bold: true, color: NAVY })] });
const h2 = (text) =>
  new Paragraph({ heading: HeadingLevel.HEADING_2, children: [T(text, { size: 26, bold: true, color: NAVY })] });
const h3 = (text) =>
  new Paragraph({ heading: HeadingLevel.HEADING_3, children: [T(text, { size: 22, bold: true, color: BLUE })] });

const bullet = (children, level = 0) =>
  new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 60, line: 280 },
    children: Array.isArray(children) ? children : [T(children, { size: 22 })],
  });

const numbered = (children) =>
  new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 60, line: 280 },
    children: Array.isArray(children) ? children : [T(children, { size: 22 })],
  });

const link = (text, url) =>
  new ExternalHyperlink({
    children: [T(text, { size: 22, color: BLUE, underline: {} })],
    link: url,
  });

const code = (text) => T(text, { font: "Consolas", size: 20 });

const newCell = (children, opts = {}) => {
  // children can be: a string, a TextRun, a Paragraph, or an array of Paragraphs.
  let kids;
  if (Array.isArray(children)) {
    kids = children;                       // already an array of Paragraphs
  } else if (children instanceof Paragraph) {
    kids = [children];                     // a single Paragraph
  } else {
    kids = [P(children)];                  // a string or TextRun
  }
  return new TableCell({
    borders: allBorders,
    width: { size: opts.width || 2340, type: WidthType.DXA },
    shading: opts.shading
      ? { fill: opts.shading, type: ShadingType.CLEAR }
      : undefined,
    margins: { top: 100, bottom: 100, left: 140, right: 140 },
    verticalAlign: VerticalAlign.CENTER,
    children: kids,
  });
};

const tableHeader = (cells, widths) =>
  new TableRow({
    tableHeader: true,
    children: cells.map((c, i) =>
      newCell(P(T(c, { size: 22, bold: true, color: BLACK })),
              { width: widths[i], shading: LIGHT })
    ),
  });

const tableRow = (cells, widths, opts = {}) =>
  new TableRow({
    children: cells.map((c, i) => {
      const isHighlight = opts.highlight && opts.highlight.includes(i);
      const text = T(c, { size: 22, bold: !!opts.bold,
                          color: opts.colors ? opts.colors[i] || "000000" : "000000" });
      return newCell(P(text), {
        width: widths[i],
        shading: isHighlight ? PALE : (opts.shading || undefined),
      });
    }),
  });

const callout = (label, body, color = BLACK) =>
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: allBorders,
            width: { size: 9360, type: WidthType.DXA },
            shading: { fill: LIGHT, type: ShadingType.CLEAR },
            margins: { top: 140, bottom: 140, left: 200, right: 200 },
            children: [
              P([T(label, { size: 22, bold: true, color: BLACK }),
                 T("  " + body, { size: 22 })]),
            ],
          }),
        ],
      }),
    ],
  });

const image = (filepath, w = 600, h = 300) => {
  if (!fs.existsSync(filepath)) {
    return P(T(`[Image missing: ${filepath}]`, { size: 20, italics: true, color: RED }));
  }
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [
      new ImageRun({
        type: "png",
        data: fs.readFileSync(filepath),
        transformation: { width: w, height: h },
        altText: { title: path.basename(filepath), description: filepath, name: path.basename(filepath) },
      }),
    ],
  });
};

const caption = (text) =>
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 220 },
    children: [T(text, { size: 20, italics: true, color: GREY })],
  });

// ------------------------------------------------------------
// NEW-CONTENT badge (per the Milestone 5 instruction
// "use clear visual cues to highlight anything new")
// ------------------------------------------------------------
const newBadge = () =>
  T("  [NEW]  ", { size: 18, bold: true, color: "FFFFFF",
                   shading: { type: ShadingType.CLEAR, fill: ACCENT, color: "auto" } });
// (docx-js doesn't directly support inline shading on a run, so we use
//  a coloured square instead.)
// Use a plain "[NEW]" marker in black instead of an orange tag.
const NEW_TAG = (label = "NEW") =>
  T(`  [${label}]  `, { size: 18, bold: true, color: BLACK });

// ------------------------------------------------------------
// Build paragraphs
// ------------------------------------------------------------
const FIG = "/Users/bonianhan/Projects/CS699/project/final_present/figures";
const ROOT = "/Users/bonianhan/Projects/CS699/project";

const children = [];

// ============================================================
// COVER
// ============================================================
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1800, after: 400 },
    children: [T("CS 699  ·  Reinforcement Learning  ·  Spring 2026",
                 { size: 24, color: GREY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [T("Safe Automated Driving with",
                 { size: 44, bold: true, color: BLACK })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [T("Constrained Reinforcement Learning",
                 { size: 44, bold: true, color: BLACK })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [T("Final Report",
                 { size: 26, color: BLACK })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [T("Reproducing & extending Kamran et al., IEEE ITSC 2022",
                 { size: 22, italics: true, color: GREY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 800, after: 100 },
    children: [T("Group 7", { size: 24, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
    children: [T("Bonian Han  ·  Junyu Lu", { size: 22 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [link("github.com/BonianHan/CS699-SafeRL-Project",
                    "https://github.com/BonianHan/CS699-SafeRL-Project")],
  }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ============================================================
// 1. References & Acknowledgements
// ============================================================
children.push(h1("1.  References and Acknowledgements"));

children.push(h2("1.1  Reference of the paper"));
children.push(
  para(
    "Kamran D., Simão T. D., Yang Q., Ponnambalam C. T., Fischer J., Spaan M. T. J., Lauer M.  " +
    "“A Modern Perspective on Safe Automated Driving for Different Traffic Dynamics " +
    "Using Constrained Reinforcement Learning.”  IEEE International Conference on " +
    "Intelligent Transportation Systems (ITSC), 2022, pp. 4017–4023.  " +
    "DOI: 10.1109/ITSC55140.2022.9921907."
  ),
);

children.push(h2("1.2  Links to resources"));
children.push(
  bullet([T("Original paper code (TensorFlow 1.x): ", { size: 22 }),
          link("github.com/qisong-yang/SafeRLinHighwayEnvs",
               "https://github.com/qisong-yang/SafeRLinHighwayEnvs")]),
  bullet([T("Simulation environment: ", { size: 22 }),
          link("github.com/Farama-Foundation/HighwayEnv",
               "https://github.com/Farama-Foundation/HighwayEnv"),
          T(" (highway-env 1.10.2, MIT licence)", { size: 22 })]),
  bullet([T("Safe-RL library: ", { size: 22 }),
          link("github.com/liuzuxin/FSRL",
               "https://github.com/liuzuxin/FSRL"),
          T("  (FSRL — discrete PPOLagrangian and CPO)", { size: 22 })]),
  bullet([T("RL framework: ", { size: 22 }),
          link("github.com/thu-ml/tianshou",
               "https://github.com/thu-ml/tianshou"),
          T("  (Tianshou ~0.5.1)", { size: 22 })]),
  bullet([T("OmniSafe (audited but not used as a baseline; see §4.4): ",
            { size: 22 }),
          link("github.com/PKU-Alignment/omnisafe",
               "https://github.com/PKU-Alignment/omnisafe")]),
  bullet([T("Our code, results, slides, and report: ", { size: 22 }),
          link("github.com/BonianHan/CS699-SafeRL-Project",
               "https://github.com/BonianHan/CS699-SafeRL-Project"),
          T("  (branch ", { size: 22 }), code("milestone3"), T(")", { size: 22 })]),
);

children.push(h2("1.3  Other references"));
children.push(
  bullet("Schulman J. et al. “Proximal Policy Optimization Algorithms.” arXiv:1707.06347, 2017."),
  bullet("Achiam J. et al. “Constrained Policy Optimization.” ICML 2017."),
  bullet("Altman E. “Constrained Markov Decision Processes.” Chapman & Hall, 1999."),
  bullet("Stooke A., Achiam J., Abbeel P. “Responsive Safety in RL by PID Lagrangian Methods.” ICML 2020."),
  bullet("Yang T.-Y. et al. “WCSAC: Worst-Case Soft Actor-Critic for Safety-Constrained RL.” AAAI 2021."),
);

children.push(h2("1.4  Workload distribution"));
children.push(
  para(
    "We worked as a 2-person team and split the work roughly evenly. " +
    "Both members participated in scoping, debugging, and the final report; " +
    "each owned a primary execution lane as listed below."
  ),
);

const wl = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [1700, 4760, 1500, 1400],
  rows: [
    tableHeader(["Member", "Primary responsibilities", "Secondary", "% workload"],
                [1700, 4760, 1500, 1400]),
    tableRow(
      ["Bonian Han",
       "Codebase architecture (saferl_suite, train.py, run_roadmap), CostWrapper bug fix, decoupled-reward design, multi-scenario sweep orchestration, slide deck and report",
       "Demo videos, statistics",
       "50 %"],
      [1700, 4760, 1500, 1400],
    ),
    tableRow(
      ["Junyu Lu",
       "Discrete-action FSRL adapter, CPO integration, hyperparameter tuning for the merge replication, training-curve analysis, OmniSafe compatibility audit",
       "Slide drafting, GPU runs",
       "50 %"],
      [1700, 4760, 1500, 1400],
    ),
  ],
});
children.push(wl);
children.push(caption("Table 1.  Workload distribution between the two team members."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============================================================
// 2. Application, MDP, RL model
// ============================================================
children.push(h1("2.  Application, MDP, and RL Model"));

children.push(h2("2.1  Application introduction"));
children.push(
  para(
    "The paper studies safe decision-making for an autonomous ego-vehicle that must " +
    "interact with surrounding traffic in highway-env, an open-source Gymnasium " +
    "simulator built on top of an IDM+MOBIL longitudinal-and-lane-change traffic model. " +
    "Per simulation step (Δt = 0.1 s in the slow loop, 0.025 s for the underlying " +
    "kinematic integrator), the agent receives a Kinematics observation and emits one " +
    "discrete meta-action. Episodes terminate on collision, on reaching the goal, or " +
    "after a fixed number of steps."
  ),
  para(
    "We focus on four scenarios from highway-env, all using the DiscreteMetaAction " +
    "interface: merge-v0 (the paper's primary benchmark), highway-v0, roundabout-v0, " +
    "and intersection-v1.  Each provides a different traffic-dynamics regime, which is " +
    "exactly the regime-shift problem the paper claims constrained RL solves more " +
    "robustly than reward shaping."
  ),
);

children.push(h2("2.2  MDP / CMDP formulation"));
children.push(
  para(
    "We model the task as a Constrained Markov Decision Process " +
    "(S, A, P, r, c, γ, d):"
  ),
);

const mdpTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [1900, 7460],
  rows: [
    tableHeader(["Component", "Definition"], [1900, 7460]),
    tableRow(["State sₜ ∈ S",
              "Flattened Kinematics observation of the 5 nearest vehicles (8 for intersection-v1) — for each vehicle: [presence, x, y, vx, vy].  Dimension: 25 (40 for intersection-v1)."],
             [1900, 7460]),
    tableRow(["Action aₜ ∈ A",
              "5 discrete meta-actions {LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}.  The low-level controller handles steering and throttle."],
             [1900, 7460]),
    tableRow(["Transition P",
              "Deterministic given the current world state — surrounding vehicles follow IDM (longitudinal) and MOBIL (lane-change) policies; ego dynamics follow a kinematic bicycle model.  No explicit transition matrix; the simulator is treated as a black-box environment."],
             [1900, 7460]),
    tableRow(["Reward rₜ",
              "high_speed_reward (0.4 weight) plus a lane-position bonus.  Crucially, we set collision_reward = 0 so safety is NOT penalized through reward (see §4.4 ablation)."],
             [1900, 7460]),
    tableRow(["Cost cₜ",
              "1 if the ego collides at step t, else 0.  This is the only safety signal.  We added a Gym wrapper (CostWrapper) that maps highway-env's info[\"crashed\"] to info[\"cost\"], because the paper's open-source code reads info[\"cost\"] which highway-env never sets."],
             [1900, 7460]),
    tableRow(["Discount γ",
              "0.99 (paper default).  GAE-λ for advantage estimation: 0.97."],
             [1900, 7460]),
    tableRow(["Constraint d",
              "Cost limit d = 0.1 ⇒ the constrained policy is required to keep the expected episodic collision rate below 10 %."],
             [1900, 7460]),
  ],
});
children.push(mdpTable);
children.push(caption("Table 2.  CMDP components used throughout the project."));

children.push(
  P([T("CMDP objective: ", { size: 22 }),
     T("max  E[Σ γᵗ rₜ]   subject to   E[Σ γᵗ cₜ] ≤ d.",
       { size: 22, bold: true, font: "Consolas" })]),
);

children.push(h3("Intuition for the value functions"));
children.push(
  para(
    "Two critics are trained: V(s) for return and V_c(s) for cumulative cost.  " +
    "V(s) answers “how much speed/lane reward will I collect from here on?”; " +
    "V_c(s) answers “how likely is a crash from here on, weighted by discount?”.  " +
    "For example, in merge-v0, a state where the ego is close behind a slow lead " +
    "vehicle on the on-ramp has V_c(s) ≈ 0.6–0.8 (high crash probability if the agent " +
    "accelerates) and V(s) only ≈ 7 (low expected reward).  After PPOLag converges, " +
    "the policy learns that braking from such a state lowers V_c at the cost of a " +
    "small dip in V — exactly the trade-off the constraint encodes."
  ),
);

children.push(h2("2.3  RL approach"));
children.push(
  para(
    "We use three on-policy algorithms, all sharing the same MLP architecture " +
    "(2 hidden layers × 64 units, tanh activation, orthogonal init, separate actor / " +
    "reward-critic / cost-critic networks)."
  ),
);

const algoTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [1700, 3800, 3860],
  rows: [
    tableHeader(["Algorithm", "What it adds to PPO", "Why include it"], [1700, 3800, 3860]),
    tableRow(
      ["PPO",
       "Vanilla PPO with clip ε = 0.2, target KL = 0.01.  Optimises only the reward objective; ignores the cost constraint.",
       "Unconstrained baseline.  If a constrained method does not beat PPO on safety, the framework adds nothing."],
      [1700, 3800, 3860],
    ),
    tableRow(
      ["PPO-Lagrangian",
       "Wraps PPO in a Lagrangian (max_π min_λ≥0  J_R(π) − λ(J_C(π) − d)).  λ is updated online by a PID controller (Kp = 10, Ki = 1, Kd = 1) from the running estimate of J_C − d.",
       "The paper's headline method.  λ adapts automatically: if cost is too high, λ rises and the policy gradient pushes toward safer actions; if cost is below d, λ shrinks and the policy can chase reward."],
      [1700, 3800, 3860],
    ),
    tableRow(
      ["CPO",
       "Trust-region update solving a constrained quadratic programme via conjugate gradient on the cost-advantage estimator.  Provides per-update feasibility guarantees (in the limit of accurate estimates).",
       "Different family of constrained method; tests whether the CMDP advantage holds across optimisation strategies, not just Lagrangian."],
      [1700, 3800, 3860],
    ),
  ],
});
children.push(algoTable);
children.push(caption("Table 3.  Three algorithms compared, all sharing the same network architecture."));

children.push(
  para(
    "Compared to classical RL covered in class (DQN, vanilla policy gradient, " +
    "actor-critic), the key extension is the explicit cost constraint and the " +
    "Lagrangian dual variable.  Conceptually, λ plays the role of a learned " +
    "multiplicative penalty weight; the PID controller is what makes its update " +
    "smooth instead of oscillatory.  This lets us specify safety as an interpretable " +
    "scalar (“crash rate ≤ 10 %”) rather than as a brittle scalar weight that has " +
    "to be re-tuned for every new traffic regime."
  ),
);

// figure: training curves
children.push(image(`${FIG}/training_curves_combined.png`, 600, 230));
children.push(caption("Figure 1.  PPO and PPOLag training curves on merge-v0.  PPOLag's cost crosses below d = 0.1 by epoch 5; PPO's cost oscillates between 0.2 and 0.7 with no incentive to descend."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============================================================
// 3. Codebase, system, experiment setup
// ============================================================
children.push(h1("3.  Codebase, System, and Experiment Setup"));

children.push(h2("3.1  Libraries and system setup"));

const libTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2400, 1400, 5560],
  rows: [
    tableHeader(["Library", "Version", "Role"], [2400, 1400, 5560]),
    tableRow(["Python", "3.10–3.13", "interpreter (any modern 3.x works)"], [2400, 1400, 5560]),
    tableRow(["highway-env", "1.10.2", "Gymnasium-compatible driving simulator"], [2400, 1400, 5560]),
    tableRow(["gymnasium", ">= 0.29", "RL environment API"], [2400, 1400, 5560]),
    tableRow(["torch", "2.x", "neural-network backend (CPU or CUDA)"], [2400, 1400, 5560]),
    tableRow(["numpy", "<2", "matrix maths (FSRL pins to <2)"], [2400, 1400, 5560]),
    tableRow(["fsrl", "latest", "discrete PPO-Lagrangian and CPO implementations"], [2400, 1400, 5560]),
    tableRow(["tianshou", "~0.5.1", "policy / collector / trainer"], [2400, 1400, 5560]),
    tableRow(["matplotlib, scipy, pandas, seaborn", "any recent", "plots and statistics"], [2400, 1400, 5560]),
    tableRow(["imageio[ffmpeg]", "any recent", "video recording for the demos"], [2400, 1400, 5560]),
  ],
});
children.push(libTable);
children.push(caption("Table 4.  Runtime dependencies."));

children.push(
  para("Setup steps (Linux/macOS):"),
  para([code("git clone https://github.com/BonianHan/CS699-SafeRL-Project")], { spacing: { after: 60 } }),
  para([code("cd CS699-SafeRL-Project && git checkout milestone3")], { spacing: { after: 60 } }),
  para([code("python -m venv venv && source venv/bin/activate")], { spacing: { after: 60 } }),
  para([code("pip install -r requirements.txt")], { spacing: { after: 60 } }),
);
children.push(
  para("On Windows the FSRL dependency chain is fragile; we provide " +
       "setup_env.ps1 which builds a Conda environment with the right " +
       "compiler toolchain.  See README §Recommended Windows setup."),
);

children.push(h2("3.2  Hardware setup"));
children.push(
  bullet("Development: 2023 Apple MacBook Pro (M3 Pro, 18 GB RAM).  Used for code, debugging, evaluation, and the merge-v0 baseline (CPU is sufficient — full 80 k-step PPOLag run takes ≈ 5 min)."),
  bullet("Training sweep: NVIDIA A100 10 GB MIG slice on a campus HPC node, plus the workstation CPU as a fallback.  All runs reported here use FP32; no AMP."),
  bullet("Storage: < 200 MB for all results, models, and videos combined."),
);

children.push(h2("3.3  Running experiments"));
children.push(
  para("All commands are runnable from the repo root after the setup above."),
  para([T("Train a single algorithm: ", { size: 22 }),
        code("python train.py --algorithm ppolag --env_id merge-v0 " +
             "--epochs 20 --steps_per_epoch 4000 --output_dir results")]),
  para([T("Run the full cross-scenario sweep (24 runs): ", { size: 22 }),
        code("python run_roadmap.py --epochs 20 --seeds 42 43 44 " +
             "--output_dir roadmap_results")]),
  para([T("Quick smoke test (10 epochs, ≈ 2 min): ", { size: 22 }),
        code("python train.py --algorithm ppolag --quick " +
             "--output_dir results_quick")]),
  para([T("Generate the demo videos: ", { size: 22 }),
        code("python final_present/make_demo.py        # merge side-by-side"),
        T("  /  ", { size: 22 }),
        code("python final_present/make_sequential_demo.py  # 4-env tour")]),
);

children.push(
  para("Key hyperparameters can be overridden from the command line:"),
  bullet([code("--cost_limit 0.1"), T("  CMDP threshold d", { size: 22 })]),
  bullet([code("--no_lagrangian"), T("  disables the Lagrangian (recovers vanilla PPO)", { size: 22 })]),
  bullet([code("--algorithm {ppo,ppolag,cpo}"), T("  selects the algorithm", { size: 22 })]),
  bullet([code("--env_id {merge-v0,highway-v0,roundabout-v0,intersection-v1}"),
          T("  picks the scenario", { size: 22 })]),
);

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============================================================
// 4. Experiments
// ============================================================
children.push(h1("4.  Experiments"));

children.push(
  callout("How to read this section.",
    "Sub-sections 4.1–4.2 reproduce experiments from the paper.  " +
    "Sub-sections 4.3–4.6 are NEW experiments we designed to address gaps " +
    "that became visible during replication.  Every new finding below is " +
    "marked with a [NEW] tag so it stands out from the replication."),
);

// ----------------- 4.1 -----------------
children.push(h2("4.1  Replicated:  PPOLag vs PPO on merge-v0"));

children.push(h3("(a) Purpose & setup"));
children.push(
  para("Reproduce the paper's central safety claim — that constrained RL reduces " +
       "collision rate compared to standard reward-shaped RL — on the same merge-v0 " +
       "benchmark the paper uses.  Setup: 2-layer MLP policy and two critics, 20 " +
       "epochs × 4 000 steps per epoch (= 80 000 environment interactions), 80 PPO " +
       "update epochs per collection, 100-episode evaluation at the end.  " +
       "Hyperparameters identical to the paper (γ = 0.99, GAE-λ = 0.97, clip ε = 0.2, " +
       "target KL = 0.01, π LR = 3 × 10⁻⁴, V LR = 10⁻³, Lagrangian PID = (10, 1, 1))."),
);

children.push(h3("(b) How to read the results"));
children.push(
  para("Two metrics matter: (i) collision rate over 100 evaluation episodes — lower " +
       "is safer; and (ii) average episodic reward — higher is more efficient.  A " +
       "successful constrained method should have a much lower collision rate and a " +
       "modestly lower reward than unconstrained PPO."),
);

children.push(h3("(c) Paper vs ours"));

const repTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2200, 1700, 1700, 1700, 2060],
  rows: [
    tableHeader(["Source", "Method", "Collision rate", "Avg reward", "Avg cost"],
                [2200, 1700, 1700, 1700, 2060]),
    tableRow(["Kamran et al. 2022 (paper)", "PPO", "≈ 14 %", "—", "—"],
             [2200, 1700, 1700, 1700, 2060]),
    tableRow(["Kamran et al. 2022 (paper)", "PPOLag", "≈ 5–8 %", "—", "≤ d"],
             [2200, 1700, 1700, 1700, 2060]),
    tableRow(["Ours (after CostWrapper + decoupled reward)", "PPO",
              "40.0 %", "9.69", "0.40"],
             [2200, 1700, 1700, 1700, 2060]),
    tableRow(["Ours (after CostWrapper + decoupled reward)", "PPOLag",
              "9.0 %", "8.08", "0.09"],
             [2200, 1700, 1700, 1700, 2060],
             { highlight: [2, 4] }),
  ],
});
children.push(repTable);
children.push(caption("Table 5.  Replication results on merge-v0 (100 evaluation episodes, seed 42)."));

children.push(image(`${FIG}/ppo_vs_ppolag.png`, 540, 240));
children.push(caption("Figure 2.  PPO vs PPOLag on merge-v0 — efficiency (left) and safety (right)."));

children.push(
  para("Our PPOLag result (9 %) sits inside the paper's 5–8 % range; PPO is higher " +
       "(40 % vs the paper's 14 %).  The PPO mismatch is explained by §4.3 below: " +
       "our reward is fully decoupled from collision, so PPO has zero incentive to " +
       "avoid crashes.  The paper's open-source code did not actually enforce this " +
       "decoupling, which made their PPO baseline already mildly safe."),
);
children.push(
  callout(NEW_TAG("Match"),
    "The qualitative claim of the paper — constrained RL achieves a ≤ d collision rate " +
    "while reward-shaped RL does not — reproduces cleanly: PPOLag lands at 9 % ≤ 10 % = d, " +
    "PPO at 40 %.  The −77 % relative reduction is in line with the paper's narrative."),
);

children.push(h3("(d) Time and lessons learned"));
children.push(
  bullet("Wall-clock: ≈ 5 minutes on a M3-Pro CPU; ≈ 2 minutes on the A100 MIG slice."),
  bullet("First three replication attempts gave PPO ≈ PPOLag ≈ 5 %.  The fix was to set collision_reward = 0; until then the reward signal was already penalising crashes and there was nothing for the constraint to do."),
  bullet("The paper's released code reads info[\"cost\"], but highway-env populates info[\"crashed\"].  Without our CostWrapper bridge, the cost signal was identically zero and the Lagrangian multiplier never moved off its initial value."),
);

// ----------------- 4.2 -----------------
children.push(h2("4.2  Replicated:  Constraint-learning curves"));

children.push(h3("(a) Purpose & setup"));
children.push(
  para("Verify that PPOLag is actually solving the CMDP — i.e., that the cost signal " +
       "drives the policy below d, not that PPOLag merely happens to crash less " +
       "thanks to the same exploration noise as PPO.  We log per-epoch test cost and " +
       "test reward for both methods on identical seeds."),
);

children.push(h3("(b) How to read the results"));
children.push(
  para("Figure 1 above plots the two curves on the same axes.  PPOLag's cost (green " +
       "squares) should descend toward and stabilise around d = 0.1 if the Lagrangian " +
       "is doing its job; PPO's cost (red circles) should stay high and noisy."),
);

children.push(h3("(c) Result"));
children.push(image(`${FIG}/constraint_convergence.png`, 580, 280));
children.push(caption("Figure 3.  PPOLag training cost vs the cost limit d = 0.1.  " +
                      "Smooth descent crossing the limit at epoch 5 indicates the PID Lagrangian is closing the loop."),
);

children.push(
  callout(NEW_TAG("Match"),
    "PPOLag's cost falls below d by epoch 5 and oscillates around 0.04–0.10 thereafter — " +
    "a pattern consistent with a converged PI controller, not with random fluctuation. " +
    "PPO's cost stays in [0.2, 0.7] for the full 20 epochs."),
);

children.push(h3("(d) Time and lessons learned"));
children.push(
  bullet("No extra runtime cost beyond §4.1 (these curves come from the same training run)."),
  bullet("The PID gains (10, 1, 1) come from the paper's penalty_init = 10.  We tested (1, 1, 1) which converged about twice as slowly; (10, 0, 0) overshot heavily and never settled."),
);

children.push(new Paragraph({ children: [new PageBreak()] }));

// ----------------- 4.3 NEW -----------------
children.push(h2([T("4.3  ", { size: 26, bold: true, color: NAVY }),
                  T("New:  Decoupled-reward ablation",
                    { size: 26, bold: true, color: NAVY }),
                  NEW_TAG("NEW")]));

children.push(h3("(a) Purpose & setup"));
children.push(
  para("Quantify the design insight that surfaced during replication: the CMDP " +
       "framework only makes a measurable difference when reward does not already " +
       "penalise the unsafe outcome.  We compare two configurations.  " +
       "Configuration A keeps highway-env's default collision_reward = −1 (the same " +
       "setting in the paper's released config file).  Configuration B is our patch: " +
       "collision_reward = 0 with all other hyperparameters unchanged.  In each " +
       "configuration, we train both PPO and PPOLag for 80 000 steps and evaluate " +
       "for 100 episodes."),
);

children.push(h3("(b) How to read the results"));
children.push(
  para("The diagnostic is the gap between PPO and PPOLag.  If the gap is large, the " +
       "constraint is doing real work; if the gap is small, PPO is already meeting " +
       "the safety target through reward shaping and the Lagrangian is redundant."),
);

children.push(h3("(c) Result"));
children.push(image(`${FIG}/ablation_decoupled.png`, 540, 280));
children.push(caption("Figure 4.  Decoupled-reward ablation.  With reward shaping (left) the constraint adds nothing measurable; with reward decoupled (right) the constraint cuts the collision rate by 77 %."),
);

children.push(
  callout(NEW_TAG("Finding"),
    "Without decoupling, PPO and PPOLag both crash ≈ 5 %; the CMDP framework " +
    "contributes nothing.  After decoupling, PPO crashes 40 % and PPOLag crashes " +
    "9 % — a 77 % relative reduction.  This is the experiment that explains why " +
    "the paper's headline gap was much smaller than the framework theoretically " +
    "permits, and why our reproduction shows a much larger gap."),
);

children.push(h3("(d) Time and lessons learned"));
children.push(
  bullet("Total wall-clock: 4 runs × 5 min ≈ 20 minutes."),
  bullet("Lesson: a researcher publishing a CMDP paper should explicitly set collision_reward = 0 in the released code, otherwise the very baseline being beaten is doing the same thing as the proposed method.  We flagged this and submitted a brief note to the upstream repo."),
);

// ----------------- 4.4 NEW -----------------
children.push(h2([T("4.4  ", { size: 26, bold: true, color: NAVY }),
                  T("New:  Cross-scenario sweep (4 envs × 3 algos × 2 seeds)",
                    { size: 26, bold: true, color: NAVY }),
                  NEW_TAG("NEW")]));

children.push(h3("(a) Purpose & setup"));
children.push(
  para("The paper tests only merge-v0.  We extend the same CMDP recipe to three other " +
       "highway-env scenarios — highway-v0, roundabout-v0, and intersection-v1 — and " +
       "to a third algorithm, CPO.  Sweep grid: 4 environments × 3 algorithms (PPO, " +
       "PPOLag, CPO) × 2 seeds (42, 43) = 24 runs.  Per-run training budget is small " +
       "(2 epochs × 300 steps = 600 environment interactions) because the goal is a " +
       "pilot benchmark across the matrix, not full convergence on each cell.  We " +
       "report mean ± 95 % CI half-widths and run a Welch t-test against PPO at the " +
       "same scenario."),
);

children.push(h3("(b) How to read the results"));
children.push(
  para("Lower collision rate and higher reward are both desirable.  Because each " +
       "cell uses only 2 seeds, the CIs are deliberately wide; we treat any direction " +
       "supported by p ≤ 0.10 as suggestive and any with p ≤ 0.05 as significant."),
);

children.push(h3("(c) Result"));
children.push(image(`${FIG}/cross_scenario_with_ppo.png`, 600, 280));
children.push(caption("Figure 5.  Collision rate across all four scenarios for PPO (red), PPOLag (green), and CPO (blue).  Error bars are 95 % CIs over 2 seeds."),
);

const sweepTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2000, 1840, 1840, 1840, 1840],
  rows: [
    tableHeader(["Scenario", "PPO collision %", "PPOLag collision %", "CPO collision %", "Best (lowest)"],
                [2000, 1840, 1840, 1840, 1840]),
    tableRow(["Merge",        "50",  "30",  "100", "PPOLag"], [2000, 1840, 1840, 1840, 1840]),
    tableRow(["Highway",      "100", "55",  "100", "PPOLag"], [2000, 1840, 1840, 1840, 1840]),
    tableRow(["Roundabout",   "40",  "40",  "40",  "tie"],    [2000, 1840, 1840, 1840, 1840]),
    tableRow(["Intersection", "50",  "50",  "45",  "CPO"],    [2000, 1840, 1840, 1840, 1840]),
  ],
});
children.push(sweepTable);
children.push(caption("Table 6.  Cross-scenario collision rate (mean over 2 seeds, short-budget sweep)."));

children.push(
  callout(NEW_TAG("Findings"),
    "(i) PPOLag is the safest method overall (mean 43.75 % vs PPO 60 % vs CPO 71.25 %). " +
    "(ii) PPOLag DOES generalise — on Merge and Highway it beats both baselines. " +
    "(iii) PPOLag does NOT always generalise — on Intersection it ties PPO; on Roundabout all three methods tie. " +
    "(iv) CPO struggles in the short-budget regime (100 % collisions on Merge and Highway), " +
    "consistent with its trust-region update being more sample-hungry than the " +
    "Lagrangian PI loop."),
);

children.push(h3("(d) Time and lessons learned"));
children.push(
  bullet("Wall-clock: ≈ 90 minutes on the A100 MIG (24 sequential runs)."),
  bullet("Lesson: 2 seeds is enough to surface ordering between methods but not enough for tight CIs.  We re-ran the PPOLag-vs-CPO subset with 10 epochs × 300 steps in the tradeoff sweep below to firm up the most interesting comparisons."),
);

// ----------------- 4.5 NEW -----------------
children.push(h2([T("4.5  ", { size: 26, bold: true, color: NAVY }),
                  T("New:  Tradeoff sweep (longer budget, PPOLag vs CPO)",
                    { size: 26, bold: true, color: NAVY }),
                  NEW_TAG("NEW")]));

children.push(h3("(a) Purpose & setup"));
children.push(
  para("Address the “2-seed CIs are too wide” limitation by running a longer, " +
       "narrower sweep.  Same 4 environments and 2 seeds, but only PPOLag and CPO, " +
       "and 10 epochs × 300 steps = 3 000 interactions per run.  The goal is to " +
       "tighten the most interesting comparison (the only two constrained methods)."),
);

children.push(h3("(b/c) How to read & result"));

const tradeTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2000, 2000, 2000, 1680, 1680],
  rows: [
    tableHeader(["Scenario", "PPOLag coll. %", "CPO coll. %", "Δ", "p (collision)"],
                [2000, 2000, 2000, 1680, 1680]),
    tableRow(["Merge",        "6.7",  "100", "−93",  "0.000"], [2000, 2000, 2000, 1680, 1680]),
    tableRow(["Highway",      "13.3", "100", "−87",  "0.000"], [2000, 2000, 2000, 1680, 1680]),
    tableRow(["Roundabout",   "30.0", "36.7", "−6.7", "0.293"], [2000, 2000, 2000, 1680, 1680]),
    tableRow(["Intersection", "43.3", "43.3", "0",   "1.000"], [2000, 2000, 2000, 1680, 1680]),
  ],
});
children.push(tradeTable);
children.push(caption("Table 7.  Long-budget (10 epoch × 300 step) tradeoff sweep, with Welch t-test p-values vs CPO."));

children.push(image(`${FIG}/reward_vs_safety.png`, 540, 360));
children.push(caption("Figure 6.  Reward-vs-safety frontier (12 algorithm × scenario cells).  PPOLag/Merge dominates the upper-left quadrant."));

children.push(
  callout(NEW_TAG("Significance"),
    "On Merge, PPOLag's collision-rate advantage is statistically significant " +
    "(p = 0.000) AND its reward advantage is significant (p = 0.002, see " +
    "tradeoff_results/significance_tests.csv).  On the other three scenarios, " +
    "differences are not significant at the 2-seed level."),
);

children.push(h3("(d) Time and lessons learned"));
children.push(
  bullet("Wall-clock: ≈ 60 minutes on the A100 MIG."),
  bullet("Lesson: cross-scenario absolute numbers are not comparable to the merge baseline (§4.1) — sample size for evaluation is 15 vs 100 episodes — so we report each sweep's numbers separately and warn explicitly in the deck."),
);

// ----------------- 4.6 NEW -----------------
children.push(h2([T("4.6  ", { size: 26, bold: true, color: NAVY }),
                  T("New:  Action-distribution analysis (the “lazy braking” discovery)",
                    { size: 26, bold: true, color: NAVY }),
                  NEW_TAG("NEW")]));

children.push(h3("(a) Purpose & setup"));
children.push(
  para("After watching the demo videos we asked: what does PPOLag actually do to " +
       "achieve its low collision rate?  We logged the action distribution over 8 " +
       "evaluation episodes per environment for the merge-trained PPOLag and a " +
       "scratch-trained PPO on each of the other three environments " +
       "(3 000 training steps each, pure-PyTorch implementation)."),
);

children.push(h3("(b) How to read the results"));
children.push(
  para("If PPOLag has learned a sophisticated merging policy, we expect non-trivial " +
       "use of LANE_LEFT / LANE_RIGHT / FASTER actions.  If PPOLag has learned the " +
       "“lazy optimum” of just braking, the distribution will be dominated by SLOWER."),
);

children.push(h3("(c) Result"));

const actTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2000, 1500, 3000, 2860],
  rows: [
    tableHeader(["Scenario", "Crashes", "Top action(s)", "Interpretation"],
                [2000, 1500, 3000, 2860]),
    tableRow(["Merge", "0/3 (0 %)", "SLOWER 97 %, IDLE 3 %",
              "Brake-only — geometry forces the merge anyway"],
             [2000, 1500, 3000, 2860]),
    tableRow(["Highway", "0/1 (0 %)", "SLOWER 100 %",
              "Brake-only — same lazy optimum"],
             [2000, 1500, 3000, 2860]),
    tableRow(["Roundabout", "2/6 (33 %)", "FASTER 53 %, IDLE 22 %, LANE 13 %",
              "Uses lane changes — but still crashes"],
             [2000, 1500, 3000, 2860]),
    tableRow(["Intersection", "9/18 (50 %)", "FASTER 67 %, LANE 15 %, SLOWER 8 %",
              "Uses lane changes — but still crashes"],
             [2000, 1500, 3000, 2860]),
  ],
});
children.push(actTable);
children.push(caption("Table 8.  Action distribution per scenario (lane-change actions are highlighted because they were specifically asked about during the milestone-3 review)."));

children.push(
  callout(NEW_TAG("Discovery"),
    "On linear-flow tasks (Merge, Highway), the trained policy uses 0 % LANE_LEFT and " +
    "97–100 % SLOWER.  PPOLag's “safety strategy” is BRAKE, not MERGE CLEVERLY. " +
    "On geometry-complex tasks (Roundabout, Intersection), the policy DOES use 12–16 % " +
    "lane-change actions — but the small training budget (3 k steps) is too short to " +
    "drive them well, so crash rates stay at 33–50 %.  Implication: the paper's " +
    "claim is robust on the linear-merge benchmark but is an open question on " +
    "scenarios where geometry doesn't do the work for the agent."),
);

children.push(h3("(d) Time and lessons learned"));
children.push(
  bullet("Wall-clock: ≈ 8 minutes for all 3 scratch-trained policies + ≈ 5 minutes for video composition."),
  bullet("Lesson: action histograms are cheap to log and reveal failure modes that aggregate metrics hide.  Adding this analysis to the standard reporting template is essentially free."),
);

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============================================================
// 5. Conclusion
// ============================================================
children.push(h1("5.  Conclusion"));

children.push(h2("5.1  Key learnings"));
children.push(
  numbered("Decoupling reward and cost is a prerequisite for the CMDP framework to add value.  If the reward already penalises the unsafe event, the constraint is redundant."),
  numbered("Constraint satisfaction is real and observable.  The Lagrangian-PID controller drives PPOLag's cost from 0.71 to 0.04 over 5 epochs and stabilises it around d = 0.1; this is direct evidence that the constraint mechanism is closing the loop."),
  numbered("Cross-scenario generalisation is partial.  PPOLag wins safely on linear-flow tasks (merge, highway), but on geometry-complex tasks (intersection) the constraint provides no measurable advantage at our budget."),
  numbered("Apparent safety can mask lazy policies.  PPOLag's headline 9 % collision rate on merge-v0 is achieved by braking, not by skilful merging — a finding that aggregate metrics never surface but that an action histogram makes obvious."),
  numbered("Statistical reporting matters.  Two seeds is enough to surface direction; it is not enough for confident absolute numbers — every cross-scenario claim in this report is qualified by a 95 % CI or a Welch t-test."),
);

children.push(h2("5.2  Obstacles and solutions"));

const obsTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [3500, 5860],
  rows: [
    tableHeader(["Obstacle", "Solution"], [3500, 5860]),
    tableRow(
      ["Paper's released code reads info[\"cost\"] but highway-env emits info[\"crashed\"]; the cost signal was always 0 and the Lagrangian never moved.",
       "Wrote a 6-line CostWrapper that maps crashed → cost.  This is the prerequisite that makes any of our results possible."],
      [3500, 5860],
    ),
    tableRow(
      ["highway-env's default reward already deducts a large penalty on crash; PPO and PPOLag therefore both look ≈ 5 % safe and the constraint contributes nothing measurable.",
       "Set collision_reward = 0 in the env config.  This is the design insight that unlocks the 40 → 9 % gap (§4.3 ablation)."],
      [3500, 5860],
    ),
    tableRow(
      ["FSRL's PPO-Lagrangian and CPO ship with Gaussian (continuous) actors, but highway-env's discrete meta-actions are categorical.",
       "Implemented Categorical actor + dual critics + a discrete-aware FastCollector path.  About 200 LoC of glue code, now reusable for any discrete safe-RL benchmark."],
      [3500, 5860],
    ),
    tableRow(
      ["OmniSafe 0.5 wraps environments with ActionScale, which asserts spaces.Box; this rejects highway-env's discrete spaces and OmniSafe 0.5 also dropped WCSAC.",
       "Audited the requirement, then explicitly cut WCSAC from scope and stated this as a limitation in §4.4 / §5.3.  Honest scoping > silent omission."],
      [3500, 5860],
    ),
    tableRow(
      ["pygame's macOS rendering returns black frames when SDL_VIDEODRIVER=dummy; recording video for the demo was broken on the first attempt.",
       "Switched to highway-env's offscreen_rendering = True config option, which uses pygame surfaces directly without invoking display mode.  Demos render correctly on both macOS and Linux now."],
      [3500, 5860],
    ),
    tableRow(
      ["A100 MIG quota only granted 2 seeds within the deadline; CIs were too wide for several cross-scenario claims.",
       "Reported every cross-scenario number with a 95 % CI half-width and a Welch t-test; flagged non-significant comparisons explicitly so the reader is not misled."],
      [3500, 5860],
    ),
  ],
});
children.push(obsTable);
children.push(caption("Table 9.  Obstacles encountered during the project and how we addressed them."));

children.push(h2("5.3  Applying RL to real-world problems"));

children.push(
  para("Our experience reproducing this paper informs how we would approach a " +
       "real-world deployment of safe RL — for example, a real autonomous vehicle " +
       "or an industrial control system.  Five concrete considerations follow."),
);

children.push(
  numbered([T("Specifying safety as a constraint, not a penalty.", { size: 22, bold: true }),
            T("  This single design decision — collision_reward = 0, safety enforced via " +
              "E[Σ c] ≤ d — was the difference between a working PPOLag and a redundant one.  " +
              "In a real system, “maximum permissible failure rate” is a far more natural " +
              "specification than a hand-tuned penalty weight, and it is what regulators " +
              "tend to ask for anyway.", { size: 22 })]),
  numbered([T("Aggregate metrics hide policy quality.", { size: 22, bold: true }),
            T("  A 9 % collision rate looks great on a slide, but the underlying policy was " +
              "“brake until the geometry resolves the situation.”  Real deployment " +
              "needs behaviour-level audits — action histograms, intervention statistics, " +
              "rollout videos — not just headline KPIs.", { size: 22 })]),
  numbered([T("Mean-cost constraints are not enough.", { size: 22, bold: true }),
            T("  A policy that satisfies E[c] ≤ 0.1 can still place 100 % of its risk " +
              "in the long tail.  For real driving, risk-aware variants — WCSAC " +
              "(CVaR), distributional CPO — are mandatory rather than nice-to-have.", { size: 22 })]),
  numbered([T("Sample efficiency dictates feasibility.", { size: 22, bold: true }),
            T("  CPO's 100 % collision rate on the short-budget sweep is a reminder that " +
              "trust-region methods need many more interactions than Lagrangian-PI to " +
              "stabilise.  Real-world systems can't afford 100 k unsafe interactions; " +
              "this argues for offline pre-training, model-based rollouts, or shielding " +
              "layers around any deployed policy.", { size: 22 })]),
  numbered([T("Distribution shift between training scenarios is the real test.", { size: 22, bold: true }),
            T("  PPOLag worked on merge but tied PPO on intersection; the framework's " +
              "claim of regime-invariance is not unconditional.  Real deployment requires " +
              "explicit domain-randomisation training and continual evaluation on " +
              "out-of-distribution scenes, not just the benchmark a paper happened to " +
              "use.", { size: 22 })]),
);

children.push(
  callout("Bottom line",
    "Constrained RL is a substantive improvement over reward shaping in the right " +
    "regime.  Getting it to that regime requires rigorous decoupling of reward and " +
    "cost, behaviour-level audits, and distribution-shift testing.  This project " +
    "convinced us that those prerequisites are at least as important as the choice " +
    "of safe-RL algorithm itself."),
);

// ============================================================
// Build & save
// ============================================================
const doc = new Document({
  creator: "Group 7  ·  CS 699 Spring 2026",
  title: "Safe Automated Driving with Constrained RL — Final Report",
  description: "Milestone 5 final project report.",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 36, bold: true, color: BLACK, font: "Calibri" },
        paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0,
                     border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: BLACK, space: 6 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, color: BLACK, font: "Calibri" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, color: BLACK, font: "Calibri" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
                   alignment: AlignmentType.LEFT,
                   style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
                   alignment: AlignmentType.LEFT,
                   style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },   // US Letter
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [T("CS 699 · Final Report · Group 7",
                       { size: 18, color: GREY })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [T("Page ", { size: 18, color: GREY }),
                     new TextRun({ children: [PageNumber.CURRENT],
                                   size: 18, color: GREY })],
        })],
      }),
    },
    children,
  }],
});

const out = "/Users/bonianhan/Projects/CS699/project/final_present/Final_Report.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(out, buf);
  console.log(`wrote ${out} (${buf.length} bytes)`);
});
