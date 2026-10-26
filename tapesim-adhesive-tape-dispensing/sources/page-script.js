
'use strict';
const DATA = {
  "models": [
    "Coh-F",
    "Coh-R",
    "Bond-F",
    "Bond-R"
  ],
  "costs": [
    {
      "turns": 2,
      "model": "Coh-F",
      "physics_mean_ms": 50.05
    },
    {
      "turns": 2,
      "model": "Bond-F",
      "physics_mean_ms": 61.96
    },
    {
      "turns": 2,
      "model": "Bond-R",
      "physics_mean_ms": 29.41
    },
    {
      "turns": 2,
      "model": "Coh-R",
      "physics_mean_ms": 37.93
    },
    {
      "turns": 8,
      "model": "Coh-F",
      "physics_mean_ms": 177.35
    },
    {
      "turns": 8,
      "model": "Bond-F",
      "physics_mean_ms": 70.9
    },
    {
      "turns": 8,
      "model": "Bond-R",
      "physics_mean_ms": 29.82
    },
    {
      "turns": 8,
      "model": "Coh-R",
      "physics_mean_ms": 29.44
    },
    {
      "turns": 16,
      "model": "Coh-F",
      "physics_mean_ms": 234.52
    },
    {
      "turns": 16,
      "model": "Bond-F",
      "physics_mean_ms": 82.91
    },
    {
      "turns": 16,
      "model": "Bond-R",
      "physics_mean_ms": 27.07
    },
    {
      "turns": 16,
      "model": "Coh-R",
      "physics_mean_ms": 29.98
    },
    {
      "turns": 32,
      "model": "Coh-F",
      "physics_mean_ms": 307.77
    },
    {
      "turns": 32,
      "model": "Bond-F",
      "physics_mean_ms": 130.43
    },
    {
      "turns": 32,
      "model": "Bond-R",
      "physics_mean_ms": 29.14
    },
    {
      "turns": 32,
      "model": "Coh-R",
      "physics_mean_ms": 36.73
    }
  ],
  "cost_scope": "Author-corrected mean physics-step costs at comparable roll motion. 32 turns: rotation ranges matched within 5%. Repeat-level uncertainty not supplied.",
  "closed_loop": [
    {
      "task": "Peel",
      "wrap_turns": null,
      "model": "Coh-F",
      "n": 100,
      "tp": 69,
      "fp": 31,
      "fn": 0,
      "tn": 0,
      "agreement": 0.69,
      "real_successes": 69,
      "sim_successes": 100,
      "real_wilson": [
        0.5937394428051057,
        0.7722030278762395
      ],
      "sim_wilson": [
        0.9630065017930143,
        1.0
      ],
      "agreement_wilson": [
        0.5937394428051057,
        0.7722030278762395
      ],
      "balanced_accuracy": 0.5,
      "progress_pairs": 100
    },
    {
      "task": "Peel",
      "wrap_turns": null,
      "model": "Bond-F",
      "n": 100,
      "tp": 67,
      "fp": 25,
      "fn": 2,
      "tn": 6,
      "agreement": 0.73,
      "real_successes": 69,
      "sim_successes": 92,
      "real_wilson": [
        0.5937394428051057,
        0.7722030278762395
      ],
      "sim_wilson": [
        0.8500189229905127,
        0.9589065385156194
      ],
      "agreement_wilson": [
        0.6356788323205499,
        0.8073041585042365
      ],
      "balanced_accuracy": 0.5822814399251987,
      "progress_pairs": 100
    },
    {
      "task": "Peel",
      "wrap_turns": null,
      "model": "Bond-R",
      "n": 100,
      "tp": 63,
      "fp": 12,
      "fn": 6,
      "tn": 19,
      "agreement": 0.82,
      "real_successes": 69,
      "sim_successes": 75,
      "real_wilson": [
        0.5937394428051057,
        0.7722030278762395
      ],
      "sim_wilson": [
        0.656955364519384,
        0.8245478863771232
      ],
      "agreement_wilson": [
        0.7333264161369926,
        0.8829977450105364
      ],
      "balanced_accuracy": 0.7629733520336606,
      "progress_pairs": 100
    },
    {
      "task": "Peel",
      "wrap_turns": null,
      "model": "Coh-R",
      "n": 100,
      "tp": 65,
      "fp": 15,
      "fn": 4,
      "tn": 16,
      "agreement": 0.81,
      "real_successes": 69,
      "sim_successes": 80,
      "real_wilson": [
        0.5937394428051057,
        0.7722030278762395
      ],
      "sim_wilson": [
        0.7111708344068411,
        0.8666330666689676
      ],
      "agreement_wilson": [
        0.7222115462093562,
        0.8748524849023126
      ],
      "balanced_accuracy": 0.7290790088826554,
      "progress_pairs": 100
    },
    {
      "task": "Stick",
      "wrap_turns": null,
      "model": "Coh-F",
      "n": 71,
      "tp": 0,
      "fp": 0,
      "fn": 50,
      "tn": 21,
      "agreement": 0.29577464788732394,
      "real_successes": 50,
      "sim_successes": 0,
      "real_wilson": [
        0.5898146180172532,
        0.7976711479868337
      ],
      "sim_wilson": [
        0.0,
        0.05132795219689034
      ],
      "agreement_wilson": [
        0.2023288520131663,
        0.4101853819827467
      ],
      "balanced_accuracy": 0.5,
      "progress_pairs": 71
    },
    {
      "task": "Stick",
      "wrap_turns": null,
      "model": "Bond-F",
      "n": 71,
      "tp": 36,
      "fp": 9,
      "fn": 14,
      "tn": 12,
      "agreement": 0.676056338028169,
      "real_successes": 50,
      "sim_successes": 45,
      "real_wilson": [
        0.5898146180172532,
        0.7976711479868337
      ],
      "sim_wilson": [
        0.5175721979631125,
        0.7362977866602549
      ],
      "agreement_wilson": [
        0.5606115705582948,
        0.7734278828935044
      ],
      "balanced_accuracy": 0.6457142857142857,
      "progress_pairs": 71
    },
    {
      "task": "Stick",
      "wrap_turns": null,
      "model": "Bond-R",
      "n": 71,
      "tp": 35,
      "fp": 14,
      "fn": 15,
      "tn": 7,
      "agreement": 0.5915492957746479,
      "real_successes": 50,
      "sim_successes": 49,
      "real_wilson": [
        0.5898146180172532,
        0.7976711479868337
      ],
      "sim_wilson": [
        0.5751598742905824,
        0.7856027354373607
      ],
      "agreement_wilson": [
        0.47538781849423417,
        0.6983126973007014
      ],
      "balanced_accuracy": 0.5166666666666666,
      "progress_pairs": 70
    },
    {
      "task": "Stick",
      "wrap_turns": null,
      "model": "Coh-R",
      "n": 71,
      "tp": 34,
      "fp": 14,
      "fn": 16,
      "tn": 7,
      "agreement": 0.5774647887323944,
      "real_successes": 50,
      "sim_successes": 48,
      "real_wilson": [
        0.5898146180172532,
        0.7976711479868337
      ],
      "sim_wilson": [
        0.5606115705582948,
        0.7734278828935044
      ],
      "agreement_wilson": [
        0.46150748022028815,
        0.6854698792985033
      ],
      "balanced_accuracy": 0.5066666666666667,
      "progress_pairs": 71
    },
    {
      "task": "Wrap",
      "wrap_turns": 1,
      "model": "Coh-F",
      "n": 55,
      "tp": 13,
      "fp": 9,
      "fn": 15,
      "tn": 18,
      "agreement": 0.5636363636363636,
      "real_successes": 28,
      "sim_successes": 22,
      "real_wilson": [
        0.380762598469722,
        0.6362322215099672
      ],
      "sim_wilson": [
        0.2811849723246312,
        0.5318720078987862
      ],
      "agreement_wilson": [
        0.43269816346414447,
        0.6862655763936809
      ],
      "balanced_accuracy": 0.5654761904761905,
      "progress_pairs": 23
    },
    {
      "task": "Wrap",
      "wrap_turns": 1,
      "model": "Bond-F",
      "n": 55,
      "tp": 15,
      "fp": 15,
      "fn": 13,
      "tn": 12,
      "agreement": 0.4909090909090909,
      "real_successes": 28,
      "sim_successes": 30,
      "real_wilson": [
        0.380762598469722,
        0.6362322215099672
      ],
      "sim_wilson": [
        0.41522690241861226,
        0.6697471974798342
      ],
      "agreement_wilson": [
        0.36376777849003283,
        0.619237401530278
      ],
      "balanced_accuracy": 0.49007936507936506,
      "progress_pairs": 30
    },
    {
      "task": "Wrap",
      "wrap_turns": 1,
      "model": "Bond-R",
      "n": 55,
      "tp": 25,
      "fp": 12,
      "fn": 3,
      "tn": 15,
      "agreement": 0.7272727272727273,
      "real_successes": 28,
      "sim_successes": 37,
      "real_wilson": [
        0.380762598469722,
        0.6362322215099672
      ],
      "sim_wilson": [
        0.5410322355657868,
        0.7818693440483107
      ],
      "agreement_wilson": [
        0.5976780188832919,
        0.8271924806089416
      ],
      "balanced_accuracy": 0.7242063492063493,
      "progress_pairs": 37
    },
    {
      "task": "Wrap",
      "wrap_turns": 1,
      "model": "Coh-R",
      "n": 55,
      "tp": 18,
      "fp": 3,
      "fn": 10,
      "tn": 24,
      "agreement": 0.7636363636363637,
      "real_successes": 28,
      "sim_successes": 21,
      "real_wilson": [
        0.380762598469722,
        0.6362322215099672
      ],
      "sim_wilson": [
        0.265159542588938,
        0.5139077976751008
      ],
      "agreement_wilson": [
        0.6365163381083228,
        0.8563334413026682
      ],
      "balanced_accuracy": 0.7658730158730158,
      "progress_pairs": 21
    },
    {
      "task": "Wrap",
      "wrap_turns": 2,
      "model": "Coh-F",
      "n": 55,
      "tp": 1,
      "fp": 6,
      "fn": 3,
      "tn": 45,
      "agreement": 0.8363636363636363,
      "real_successes": 4,
      "sim_successes": 7,
      "real_wilson": [
        0.02864406023304053,
        0.17259940072156052
      ],
      "sim_wilson": [
        0.06304210427364634,
        0.24017027655909076
      ],
      "agreement_wilson": [
        0.7173625426094283,
        0.9114457966390772
      ],
      "balanced_accuracy": 0.5661764705882353,
      "progress_pairs": 23
    },
    {
      "task": "Wrap",
      "wrap_turns": 2,
      "model": "Bond-F",
      "n": 55,
      "tp": 0,
      "fp": 11,
      "fn": 4,
      "tn": 40,
      "agreement": 0.7272727272727273,
      "real_successes": 4,
      "sim_successes": 11,
      "real_wilson": [
        0.02864406023304053,
        0.17259940072156052
      ],
      "sim_wilson": [
        0.11552214909761245,
        0.32364879157263937
      ],
      "agreement_wilson": [
        0.5976780188832919,
        0.8271924806089416
      ],
      "balanced_accuracy": 0.39215686274509803,
      "progress_pairs": 30
    },
    {
      "task": "Wrap",
      "wrap_turns": 2,
      "model": "Bond-R",
      "n": 55,
      "tp": 1,
      "fp": 20,
      "fn": 3,
      "tn": 31,
      "agreement": 0.5818181818181818,
      "real_successes": 4,
      "sim_successes": 21,
      "real_wilson": [
        0.02864406023304053,
        0.17259940072156052
      ],
      "sim_wilson": [
        0.265159542588938,
        0.5139077976751008
      ],
      "agreement_wilson": [
        0.45033103735230756,
        0.7026223424648965
      ],
      "balanced_accuracy": 0.42892156862745096,
      "progress_pairs": 37
    },
    {
      "task": "Wrap",
      "wrap_turns": 2,
      "model": "Coh-R",
      "n": 55,
      "tp": 1,
      "fp": 3,
      "fn": 3,
      "tn": 48,
      "agreement": 0.8909090909090909,
      "real_successes": 4,
      "sim_successes": 4,
      "real_wilson": [
        0.02864406023304053,
        0.17259940072156052
      ],
      "sim_wilson": [
        0.02864406023304053,
        0.17259940072156052
      ],
      "agreement_wilson": [
        0.7817445091344648,
        0.9490327499921767
      ],
      "balanced_accuracy": 0.5955882352941176,
      "progress_pairs": 21
    }
  ],
  "replay": {
    "episodes": {
      "stick_tape_on_stand__017": {
        "Coh-F": {
          "full_mean_px": 13.14392917424255,
          "active_mean_px": 21.64831948106937
        },
        "Bond-F": {
          "full_mean_px": 15.37554434022871,
          "active_mean_px": 25.43945140070197
        },
        "Bond-R": {
          "full_mean_px": 18.374754953364047,
          "active_mean_px": 30.540333422643673
        },
        "Coh-R": {
          "full_mean_px": 16.869652564953952,
          "active_mean_px": 27.984373060099898
        }
      },
      "stick_tape_on_stand__018": {
        "Coh-F": {
          "full_mean_px": 20.400811555095643,
          "active_mean_px": 28.040436430624254
        },
        "Bond-F": {
          "full_mean_px": 10.095938560605827,
          "active_mean_px": 13.54913916612282
        },
        "Bond-R": {
          "full_mean_px": 6.6251734150749115,
          "active_mean_px": 8.649997970560644
        },
        "Coh-R": {
          "full_mean_px": 9.824938643606286,
          "active_mean_px": 13.189674162153251
        }
      },
      "stick_tape_on_stand__019": {
        "Coh-F": {
          "full_mean_px": 13.407459228754412,
          "active_mean_px": 19.035479171440876
        },
        "Bond-F": {
          "full_mean_px": 8.33354034175503,
          "active_mean_px": 11.488394256490867
        },
        "Bond-R": {
          "full_mean_px": 9.79781362066578,
          "active_mean_px": 13.636075189399335
        },
        "Coh-R": {
          "full_mean_px": 8.561255076020487,
          "active_mean_px": 11.83094473175775
        }
      },
      "stick_tape_on_stand__023": {
        "Coh-F": {
          "full_mean_px": 17.019766983197904,
          "active_mean_px": 23.396302910228513
        },
        "Bond-F": {
          "full_mean_px": 13.465490249924995,
          "active_mean_px": 18.303531826402036
        },
        "Bond-R": {
          "full_mean_px": 11.630031740156298,
          "active_mean_px": 15.712500030486616
        },
        "Coh-R": {
          "full_mean_px": 14.412109148505865,
          "active_mean_px": 19.64446757399699
        }
      },
      "stick_tape_on_stand__025": {
        "Coh-F": {
          "full_mean_px": 16.81142461322417,
          "active_mean_px": 22.676177730191583
        },
        "Bond-F": {
          "full_mean_px": 12.202969091564666,
          "active_mean_px": 16.34574329980079
        },
        "Bond-R": {
          "full_mean_px": 9.658285692604641,
          "active_mean_px": 12.864059593039254
        },
        "Coh-R": {
          "full_mean_px": 11.349184544249095,
          "active_mean_px": 15.206893927624126
        }
      }
    },
    "models": {
      "Coh-F": {
        "full_mean_px": 16.156678310902937,
        "active_mean_px": 22.95934314471092,
        "active_bootstrap95": [
          20.603183357217972,
          25.833186336634128
        ]
      },
      "Bond-F": {
        "full_mean_px": 11.894696516815847,
        "active_mean_px": 17.025251989903698,
        "active_bootstrap95": [
          13.263570752399492,
          21.634205038926154
        ]
      },
      "Bond-R": {
        "full_mean_px": 11.217211884373135,
        "active_mean_px": 16.280593241225905,
        "active_bootstrap95": [
          11.059713826313576,
          23.778630129345938
        ]
      },
      "Coh-R": {
        "full_mean_px": 12.203427995467138,
        "active_mean_px": 17.571270691126404,
        "active_bootstrap95": [
          13.049626343089226,
          23.085706297210887
        ]
      }
    },
    "bond_r_minus_bond_f": {
      "mean_px": -0.7446587486777922,
      "bootstrap95": [
        -3.8705363201125693,
        2.7937286583944085
      ]
    },
    "protocol": "Original reference time >=3 s; repeats averaged within episode; five episodes weighted equally. Exact 5^5 paired episode bootstrap, linearly interpolated 2.5/97.5 percentiles; descriptive small-n uncertainty, not annotation or alignment uncertainty."
  },
  "provenance": {
    "statistics.json": "08e9e7bf9e516c936568589a3b231b342a90f742304bce182c17b632ce210132",
    "corrected_cost.json": "e46159be05572538f4cc2f8481818a937841f79c458478664860906b8d974f58",
    "tolerance_recovery.json": "4976e1b3b5f7d23d18f73fb8af751318f7003e4a6f030957d2772084fb070566"
  },
  "fixed_tolerance_32": {
    "Coh-F": null,
    "Coh-R": 116,
    "Bond-F": 366,
    "Bond-R": 115,
    "tolerance_m_s": 0.001,
    "note": "来自论文正文；Coh-F 仅报告约 3.4x 加速，未补造精确成本"
  },
  "adhesion": {
    "factors": [
      1,
      20,
      100
    ],
    "n": 98,
    "real_successes": 68,
    "rates_pct": {
      "Coh-F": [
        100,
        70.4,
        8.2
      ],
      "Bond-F": [
        91.8,
        44.9,
        1.0
      ],
      "Bond-R": [
        75.5,
        40.8,
        0
      ],
      "Coh-R": [
        80.6,
        36.7,
        0
      ]
    },
    "source": "论文 Fig.7 标签；百分比为原文四舍五入值"
  },
  "packaging": {
    "steps": 3846,
    "dt_s": 0.03333333333333333,
    "sim_time_s": 128.2,
    "wall_time_s": 304.1,
    "reported_real_time_factor": 0.422,
    "gpu": "RTX 5090",
    "model": "Coh-R"
  },
  "demo": {
    "scope": "CPU 教学验证；不是 TapeSim 物理或性能复现",
    "paper_parameters": {
      "stiffness_pa": 1000000.0,
      "break_force_n": 0.5,
      "dt_s": 0.01,
      "front_delta_m": 0.0009,
      "collar_edges": 10
    },
    "synthetic_settings": {
      "triangle_edge_m": 0.002,
      "reference_height_m": 0.0002,
      "material_rows": 40,
      "row_length_m": 0.001
    },
    "checks": [
      {
        "name": "共同刚体变换不产生 bond 应变",
        "passed": true,
        "energy": 2.3464006867896548e-36
      },
      {
        "name": "解析梯度与独立中心差分一致",
        "passed": true,
        "maximum_relative_error": 1.9873948215709095e-09
      },
      {
        "name": "四顶点合力为零",
        "passed": true
      },
      {
        "name": "恢复力随整体旋转协变",
        "passed": true
      },
      {
        "name": "物理力不含额外时间步缩放",
        "passed": true
      },
      {
        "name": "超过力阈值的压缩仍不触发拉伸断裂",
        "passed": true,
        "force_n": 1.0000000000000016
      },
      {
        "name": "张开且超过阈值才断裂",
        "passed": true,
        "force_n": 5.000000000000012
      },
      {
        "name": "参考态不释放",
        "passed": true
      },
      {
        "name": "单独的正交势能也允许反射",
        "passed": true,
        "note": "数学边界，不表示作者系统实际发生翻转"
      },
      {
        "name": "拒绝共面参考四面体",
        "passed": true
      },
      {
        "name": "一次更新只接受旧自由区的邻居",
        "passed": true,
        "released": [
          1
        ]
      },
      {
        "name": "collar 推进会恢复下一批材料",
        "passed": true
      },
      {
        "name": "恢复原黏附历史而非重置为一",
        "passed": true
      },
      {
        "name": "第 2 次更新保持材料与质量",
        "passed": true
      },
      {
        "name": "第 3 次更新保持材料与质量",
        "passed": true
      },
      {
        "name": "第 4 次更新保持材料与质量",
        "passed": true
      },
      {
        "name": "第 5 次更新保持材料与质量",
        "passed": true
      },
      {
        "name": "第 6 次更新保持材料与质量",
        "passed": true
      },
      {
        "name": "第 7 次更新保持材料与质量",
        "passed": true
      },
      {
        "name": "第 8 次更新保持材料与质量",
        "passed": true
      },
      {
        "name": "第 9 次更新保持材料与质量",
        "passed": true
      },
      {
        "name": "没有新门槛事件时不回卷或继续前进",
        "passed": true
      },
      {
        "name": "live support 阻止 Bond 候选释放",
        "passed": true
      },
      {
        "name": "Bond 需要明确 release event",
        "passed": true
      },
      {
        "name": "Coh 几何前沿不要求 beta 已为零",
        "passed": true
      },
      {
        "name": "刚体速度投影和恢复保持运动",
        "passed": true,
        "maximum_error_m_s": 1.3877787807814457e-17
      },
      {
        "name": "表 I 的单位长度质量可由独立单位计算恢复",
        "passed": true,
        "grams_per_m": 4.446000000000001
      }
    ],
    "front_trace": [
      {
        "step": 2,
        "free": 3,
        "collar": 10,
        "cluster": 27,
        "released": [
          2
        ],
        "thawed": [
          12
        ]
      },
      {
        "step": 3,
        "free": 4,
        "collar": 10,
        "cluster": 26,
        "released": [
          3
        ],
        "thawed": [
          13
        ]
      },
      {
        "step": 4,
        "free": 5,
        "collar": 10,
        "cluster": 25,
        "released": [
          4
        ],
        "thawed": [
          14
        ]
      },
      {
        "step": 5,
        "free": 6,
        "collar": 10,
        "cluster": 24,
        "released": [
          5
        ],
        "thawed": [
          15
        ]
      },
      {
        "step": 6,
        "free": 7,
        "collar": 10,
        "cluster": 23,
        "released": [
          6
        ],
        "thawed": [
          16
        ]
      },
      {
        "step": 7,
        "free": 8,
        "collar": 10,
        "cluster": 22,
        "released": [
          7
        ],
        "thawed": [
          17
        ]
      },
      {
        "step": 8,
        "free": 9,
        "collar": 10,
        "cluster": 21,
        "released": [
          8
        ],
        "thawed": [
          18
        ]
      },
      {
        "step": 9,
        "free": 10,
        "collar": 10,
        "cluster": 20,
        "released": [
          9
        ],
        "thawed": [
          19
        ]
      }
    ],
    "limitations": [
      "没有薄壳、IPC/CCD 或隐式运动求解",
      "行图并非三角网格前沿实现",
      "没有执行真实放带、切割、GPU 计时或机器人策略"
    ]
  }
}
;
const $ = id => document.getElementById(id);
const COLORS = {'Coh-F':'#64716d','Coh-R':'#146b91','Bond-F':'#c58a1a','Bond-R':'#ad4e86',fail:'#97a9a2'};
const MODELS = ['Coh-F','Coh-R','Bond-F','Bond-R'];
const pct = v => (100*v).toFixed(1)+'%';
function bars(target, rows, max, unit, chosen='') {
  $(target).innerHTML = rows.map(r => `<div class="barrow ${r.name===chosen?'chosen':''}"><span>${r.name==='fail'?'全失败':r.name}</span><div class="track"><div class="fill" style="width:${r.value===null?0:Math.max(0,Math.min(100,100*r.value/max))}%;background:${COLORS[r.name]}"></div></div><span class="value">${r.value===null?'未列精确值':r.value.toFixed(unit==='ms'?2:1)+' '+unit}</span></div>`).join('');
}
document.querySelectorAll('[data-panel]').forEach(button => button.addEventListener('click',()=>{
 document.querySelectorAll('[data-panel]').forEach(b=>b.setAttribute('aria-selected',String(b===button)));
 document.querySelectorAll('section[role=tabpanel]').forEach(p=>p.hidden=p.id!==button.dataset.panel);
}));
const steps = [
 '运动求解：保持 bond 集合、黏附历史和 cluster 成员不变，最小化隐式增量势能。Newton/PCG 的停止精度会影响运动恢复。',
 '更新界面：求解完成后更新 β，或创建/释放 bond。断开的源对从下一次搜索恢复普通接触处理。',
 '检测前沿：只看旧自由区的一条图边外候选；位移门槛满足后，Bond 还需要释放事件及无 live support。同一次更新不能递归穿过 collar。',
 '补充 collar：相对新自由区重算图距离，把下一批 cluster 元素恢复成薄壳，保留 rest geometry 和原界面历史，继承刚体速度。',
 '可选切割：局部压力连续两步达到阈值后，在步边界移除有限宽 kerf 并重建网格、碰撞与抓取约束。例中阈值 1 kPa 是配置值。'
];
document.querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-step]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));$('step-explanation').textContent=steps[Number(b.dataset.step)];}));
$('step-explanation').textContent=steps[0];
function drawFront(){
 const progress=Number($('payout').value), free=2+progress, collar=10, cluster=42-free-collar;
 $('payout-value').textContent=`示意第 ${progress} 次推进`;
 let path='',lastX=0,lastY=0; for(let i=0;i<230;i++){let a=i/229*Math.PI*5.3,r=17+i/229*(87-progress*.7);let x=443+r*Math.cos(a),y=142+r*Math.sin(a);lastX=x;lastY=y;path+=(i?'L':'M')+x.toFixed(2)+','+y.toFixed(2);}
 let cells='';for(let i=0;i<42;i++){let color=i<free?COLORS['Coh-R']:i<free+collar?COLORS['Bond-R']:COLORS['Bond-F'];cells+=`<rect x="${22+i*13.6}" y="266" width="12.6" height="22" rx="2" fill="${color}"/>`;}
 $('roll-diagram').innerHTML=`<path d="${path}" fill="none" stroke="#c58a1a" stroke-width="13"/><circle cx="443" cy="142" r="20" fill="#f4f2ed" stroke="#64716d" stroke-width="3"/><path d="M ${lastX} ${lastY} Q 340 52 310 62" fill="none" stroke="#ad4e86" stroke-width="15"/><path d="M310 62 C235 ${70+progress*2} 188 195 75 194 L27 194" fill="none" stroke="#146b91" stroke-width="13"/><path d="M24 220 L181 220" stroke="#d8dfd9" stroke-width="13"/><path d="M49 203 L49 216 M85 203 L85 216 M121 201 L121 216" stroke="#c58a1a" stroke-width="3"/><text x="417" y="142" font-size="12">hub</text><text x="373" y="39" font-size="17">cluster · 6 DOF</text><text x="224" y="34" font-size="15">collar</text><text x="28" y="151" font-size="16">已释放：可再次黏附</text>${cells}<text x="22" y="316" font-size="12">材料坐标示意：自由区 ${free} 行 / collar ${collar} 行 / cluster ${cluster} 行</text>`;
 $('front-explanation').textContent=progress===0?'起始时，只有自由端和前沿附近保留独立变形。拖动滑块，观察自由区增长、cluster 缩小。':`示意推进 ${progress} 次后，更内侧的材料恢复为薄壳。图中的 ${collar} 行表示持续存在的局部变形区，不是论文的网格规模。`;
}
$('payout').addEventListener('input',drawFront);drawFront();
let interfaceFamily='Bond', loadState='rest';
function drawInterface(){
 const bond=interfaceFamily==='Bond';
 $('interface-explanation').textContent=bond?'对选定 point–triangle 对，以一个形状保持 bond 替换该对的 barrier、friction 和 cohesive 项。其他接触对继续工作，断裂后恢复源对普通搜索。':'使用正常接触 barrier、摩擦和带 β 历史的黏附能量。压缩促进黏合，张开或滑移促进脱黏；运动求解后更新 β。';
 $('interface-formula').innerHTML=bond?'E<sub>b</sub> = Δt² V κ ‖F Fᵀ − I‖²<br>F = D<sub>s</sub> D<sub>b</sub>⁻¹':'E<sub>a</sub> = Δt² β² [C<sub>n</sub>/(2d̂) · (d−d*)²<br>+ C<sub>t</sub>/d̂ · ‖Δx<sub>t</sub>‖²]';
 const y=loadState==='compression'?116:loadState==='opening'?20:72;
 const released=bond&&loadState==='opening';
 $('bond-diagram').innerHTML=`<polygon points="65,157 284,157 185,119" fill="#c6d7de" stroke="#64716d" stroke-width="2"/><path d="M65 157 L170 ${y} L284 157 M185 119 L170 ${y}" fill="none" stroke="${released?'#b8583e':'#ad4e86'}" stroke-width="3" ${released?'stroke-dasharray="6 6"':''}/><circle cx="170" cy="${y}" r="6" fill="#1d3438"/><text x="186" y="${y+4}" font-size="14">P</text><text x="61" y="180" font-size="13">T₀</text><text x="278" y="180" font-size="13">T₁</text><text x="194" y="117" font-size="13">T₂</text>`;
 const descriptions=bond?{compression:'压缩：d ≤ d_ref，即使恢复力模超过阈值，也不触发这条拉伸断裂判据。',rest:'参考态：相对形状不变，bond 能量为零。共同平移或旋转同样不产生应变。',opening:'张开并超过阈值：d > d_ref 且 ‖f_b‖ > f_break，步末解除 bond。物理力由 E_b / Δt² 求导。'}:{compression:'压缩：在接触带内，压力项可促进 β 增长。这里只展示状态，不数值积分 β。',rest:'参考间隙且无滑移：黏附的法向与切向弹性项为零；接触 barrier 是另一个项。',opening:'张开：随 β 的历史演化逐步脱黏。Cohesive 接口不使用 bond 的二值力阈值。'};
 $('load-explanation').textContent=descriptions[loadState];
}
document.querySelectorAll('[data-interface]').forEach(b=>b.addEventListener('click',()=>{interfaceFamily=b.dataset.interface;document.querySelectorAll('[data-interface]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));drawInterface();}));
document.querySelectorAll('[data-load]').forEach(b=>b.addEventListener('click',()=>{loadState=b.dataset.load;document.querySelectorAll('[data-load]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));drawInterface();}));drawInterface();
function drawCosts(){
 const matched=$('protocol').value==='matched';$('turns').disabled=!matched;if(!matched)$('turns').value='32';
 const turns=Number($('turns').value);
 const rows=MODELS.map(name=>({name,value:matched?DATA.costs.find(x=>x.model===name&&x.turns===turns).physics_mean_ms:DATA.fixed_tolerance_32[name]}));
 bars('cost-bars',rows,Math.max(...rows.map(x=>x.value||0)), 'ms');
 const val=name=>rows.find(x=>x.name===name).value;
 $('coh-speed').textContent=matched?(val('Coh-F')/val('Coh-R')).toFixed(2)+'×':'约 3.4×';
 $('bond-speed').textContent=matched?(val('Bond-F')/val('Bond-R')).toFixed(2)+'×':'约 3.2×';
 $('cost-protocol').textContent=matched?`${turns} 圈：以 Bond-R 的卷芯角度范围为目标，各模型选择容差，范围误差 ≤5%。下列为作者公布的均值，不是同一停止容差。`:'32 圈：Newton tolerance = 10⁻³ m/s。仅使用正文明确给出的步时；Coh-F 没有列精确值，因此不从图补造。';
}
$('protocol').addEventListener('change',drawCosts);$('turns').addEventListener('change',drawCosts);drawCosts();
function drawReplay(){
 const key=$('replay-episode').value;const source=key==='mean'?DATA.replay.models:DATA.replay.episodes[key];
 const rows=MODELS.map(name=>({name,value:source[name].active_mean_px}));bars('replay-bars',rows,Math.max(...rows.map(x=>x.value)), 'px');
 const base=source['Coh-F'].active_mean_px;
 const delta=name=>100*(base-source[name].active_mean_px)/base;
 $('replay-explanation').textContent=key==='mean'?`五段平均：Coh-R / Bond-R 相对 Coh-F 误差下降 ${delta('Coh-R').toFixed(1)}% / ${delta('Bond-R').toFixed(1)}%。这是跨 episode 平均。`:key.endsWith('017')?'Stick 17 中 clustered 模型的均值误差更大。平均改善没有保证每个 episode 都改善。':`本段：Coh-R / Bond-R 相对 Coh-F 的误差降幅为 ${delta('Coh-R').toFixed(1)}% / ${delta('Bond-R').toFixed(1)}%。`;
}
$('replay-episode').addEventListener('change',drawReplay);drawReplay();
function drawAdhesion(){
 const index=Number($('adhesion').value);bars('adhesion-bars',MODELS.map(name=>({name,value:DATA.adhesion.rates_pct[name][index]})),100,'%');
 $('adhesion-explanation').textContent=[
 '标定值下，真实成功率为 69.4%。四种模型的桌面参数相同，卷内表示与界面不同。',
 '20× 时 Coh-F 的 70.4% 偶然接近真实 69.4%，但 clustered 模型偏差扩大。仅匹配总成功率不能辨识正确黏附参数。',
 '100× 时两个 clustered 模型全部失败。总成功率对物理参数敏感，参数应先由独立剥离力测量约束。'][index];
}
$('adhesion').addEventListener('change',drawAdhesion);drawAdhesion();
let metric='sim';
function taskRows(){let key=$('task').value;return DATA.closed_loop.filter(x=>key==='Peel'||key==='Stick'?x.task===key:x.task==='Wrap'&&x.wrap_turns===Number(key.slice(-1)));}
function rates(row){return {sim:(row.tp+row.fp)/row.n,agreement:(row.tp+row.tn)/row.n,ba:0.5*(row.tp/(row.tp+row.fn)+row.tn/(row.tn+row.fp))};}
function drawPrediction(){
 const rows=taskRows(), first=rows[0], real=first.tp+first.fn, negative=first.n-real;
 const baseline={model:'fail',n:first.n,tp:0,fp:0,fn:real,tn:negative};
 const all=MODELS.map(name=>rows.find(x=>x.model===name)).concat([baseline]);
 const selected=$('model').value, row=all.find(x=>x.model===selected), values=rates(row);
 bars('prediction-bars',all.map(r=>({name:r.model,value:rates(r)[metric]*100})),100,'%',selected);
 $('task-context').textContent=`${first.n} 个配对案例；真实成功 ${real}/${first.n} = ${pct(real/first.n)}。每个域运行同一冻结策略，根据各自观测继续决策。`;
 $('matrix-title').textContent=(selected==='fail'?'恒定预测失败':selected)+' · 逐例结果';
 $('matrix').innerHTML=[['TP',row.tp,'两域都成功',false],['FP',row.fp,'仿真成功，真实失败',true],['FN',row.fn,'真实成功，仿真失败',true],['TN',row.tn,'两域都失败',false]].map(([label,value,note,bad])=>`<div class="cell ${bad?'bad':''}"><small>${label}</small><strong>${value}</strong><small>${note}</small></div>`).join('');
 $('matrix-result').textContent=`Agreement ${pct(values.agreement)} · BA ${pct(values.ba)}。真实成功召回 ${row.tp}/${real}，真实失败召回 ${row.tn}/${negative}。`;
 $('metric-explanation').textContent=metric==='sim'?`真实成功率 ${pct(real/first.n)}。本图只比较成功数量；是否为同一案例，需要查看混淆矩阵。`:metric==='agreement'?`全失败基线 agreement 为 ${pct(negative/first.n)}。类别不平衡会让这个指标偏高。`:'BA 给予真实成功和真实失败同等权重。恒定预测的 50% 是判别能力基线。';
 const key=$('task').value;
 $('prediction-warning').textContent={Peel:'Peel：clustered 两种配置的 BA 从 Coh-F 的 50% 提升到 72.9% / 76.3%。注意这是成败预测，不是新策略的真机成功率。',Stick:'Stick：Coh-R / Bond-R 总成功率接近真实 70.4%，BA 却只有 50.7% / 51.7%。Bond-F 的 BA 为 64.6%，界面与表示的排名随指标改变。',Wrap1:'Wrap ≥1 圈：Coh-R 的 BA 为 76.6%，Bond-R 为 72.4%。更快的界面配置不自动更擅长预测结果。',Wrap2:'Wrap ≥2 圈：Coh-R 仿真与真实均成功 4/55，但共同成功只有 1 次。其 89.1% agreement 低于全失败基线 92.7%；BA 为 59.6%。稀少成功仍难预测。'}[key];
}
$('task').addEventListener('change',drawPrediction);$('model').addEventListener('change',drawPrediction);
document.querySelectorAll('[data-metric]').forEach(b=>b.addEventListener('click',()=>{metric=b.dataset.metric;document.querySelectorAll('[data-metric]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));drawPrediction();}));drawPrediction();
const checks=DATA.demo.checks;
const grad=checks.find(x=>x.maximum_relative_error!==undefined).maximum_relative_error;
$('demo-status').textContent=`CPU 教学 demo 已通过 ${checks.length} 项行为检查。bond 解析梯度与中心差分的最大归一化误差约 ${grad.toExponential(2)}。`;
const DEEP = {"math":{"reference_volume_m3":1.3333333333333354e-10,"stiffness_pa":1000000.0,"triangle_area_m2":2e-06,"critical_normal_stretch":1.0831036969405674,"break_force_n":0.5,"hessian_error":4.599793988389357e-10},"mesh":{"datasets":{"3":{"rows":31,"across":3,"rest_m":[[0.0,-0.0095,0.0],[0.0,0.0,0.0],[0.0,0.0095,0.0],[0.002,-0.0095,0.0],[0.002,0.0,0.0],[0.002,0.0095,0.0],[0.004,-0.0095,0.0],[0.004,0.0,0.0],[0.004,0.0095,0.0],[0.006,-0.0095,0.0],[0.006,0.0,0.0],[0.006,0.0095,0.0],[0.008,-0.0095,0.0],[0.008,0.0,0.0],[0.008,0.0095,0.0],[0.01,-0.0095,0.0],[0.01,0.0,0.0],[0.01,0.0095,0.0],[0.012,-0.0095,0.0],[0.012,0.0,0.0],[0.012,0.0095,0.0],[0.014,-0.0095,0.0],[0.014,0.0,0.0],[0.014,0.0095,0.0],[0.016,-0.0095,0.0],[0.016,0.0,0.0],[0.016,0.0095,0.0],[0.018000000000000002,-0.0095,0.0],[0.018000000000000002,0.0,0.0],[0.018000000000000002,0.0095,0.0],[0.02,-0.0095,0.0],[0.02,0.0,0.0],[0.02,0.0095,0.0],[0.022,-0.0095,0.0],[0.022,0.0,0.0],[0.022,0.0095,0.0],[0.024,-0.0095,0.0],[0.024,0.0,0.0],[0.024,0.0095,0.0],[0.026000000000000002,-0.0095,0.0],[0.026000000000000002,0.0,0.0],[0.026000000000000002,0.0095,0.0],[0.028,-0.0095,0.0],[0.028,0.0,0.0],[0.028,0.0095,0.0],[0.03,-0.0095,0.0],[0.03,0.0,0.0],[0.03,0.0095,0.0],[0.032,-0.0095,0.0],[0.032,0.0,0.0],[0.032,0.0095,0.0],[0.034,-0.0095,0.0],[0.034,0.0,0.0],[0.034,0.0095,0.0],[0.036000000000000004,-0.0095,0.0],[0.036000000000000004,0.0,0.0],[0.036000000000000004,0.0095,0.0],[0.038,-0.0095,0.0],[0.038,0.0,0.0],[0.038,0.0095,0.0],[0.04,-0.0095,0.0],[0.04,0.0,0.0],[0.04,0.0095,0.0],[0.042,-0.0095,0.0],[0.042,0.0,0.0],[0.042,0.0095,0.0],[0.044,-0.0095,0.0],[0.044,0.0,0.0],[0.044,0.0095,0.0],[0.046,-0.0095,0.0],[0.046,0.0,0.0],[0.046,0.0095,0.0],[0.048,-0.0095,0.0],[0.048,0.0,0.0],[0.048,0.0095,0.0],[0.05,-0.0095,0.0],[0.05,0.0,0.0],[0.05,0.0095,0.0],[0.052000000000000005,-0.0095,0.0],[0.052000000000000005,0.0,0.0],[0.052000000000000005,0.0095,0.0],[0.054,-0.0095,0.0],[0.054,0.0,0.0],[0.054,0.0095,0.0],[0.056,-0.0095,0.0],[0.056,0.0,0.0],[0.056,0.0095,0.0],[0.058,-0.0095,0.0],[0.058,0.0,0.0],[0.058,0.0095,0.0],[0.06,-0.0095,0.0],[0.06,0.0,0.0],[0.06,0.0095,0.0]],"triangles":[[0,3,1],[1,3,4],[1,4,2],[2,4,5],[3,6,4],[4,6,7],[4,7,5],[5,7,8],[6,9,7],[7,9,10],[7,10,8],[8,10,11],[9,12,10],[10,12,13],[10,13,11],[11,13,14],[12,15,13],[13,15,16],[13,16,14],[14,16,17],[15,18,16],[16,18,19],[16,19,17],[17,19,20],[18,21,19],[19,21,22],[19,22,20],[20,22,23],[21,24,22],[22,24,25],[22,25,23],[23,25,26],[24,27,25],[25,27,28],[25,28,26],[26,28,29],[27,30,28],[28,30,31],[28,31,29],[29,31,32],[30,33,31],[31,33,34],[31,34,32],[32,34,35],[33,36,34],[34,36,37],[34,37,35],[35,37,38],[36,39,37],[37,39,40],[37,40,38],[38,40,41],[39,42,40],[40,42,43],[40,43,41],[41,43,44],[42,45,43],[43,45,46],[43,46,44],[44,46,47],[45,48,46],[46,48,49],[46,49,47],[47,49,50],[48,51,49],[49,51,52],[49,52,50],[50,52,53],[51,54,52],[52,54,55],[52,55,53],[53,55,56],[54,57,55],[55,57,58],[55,58,56],[56,58,59],[57,60,58],[58,60,61],[58,61,59],[59,61,62],[60,63,61],[61,63,64],[61,64,62],[62,64,65],[63,66,64],[64,66,67],[64,67,65],[65,67,68],[66,69,67],[67,69,70],[67,70,68],[68,70,71],[69,72,70],[70,72,73],[70,73,71],[71,73,74],[72,75,73],[73,75,76],[73,76,74],[74,76,77],[75,78,76],[76,78,79],[76,79,77],[77,79,80],[78,81,79],[79,81,82],[79,82,80],[80,82,83],[81,84,82],[82,84,85],[82,85,83],[83,85,86],[84,87,85],[85,87,88],[85,88,86],[86,88,89],[87,90,88],[88,90,91],[88,91,89],[89,91,92]],"states":[{"step":0,"n":3,"free":[0,1,2],"rigid":[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[3,4,5,6,7,8],"cluster_elements":[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.24008399999999996,"shell_mass_g":0.026676,"independent_dof":33,"update":null},{"step":1,"n":3,"free":[0,1,2,3,4,5],"rigid":[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[6,7,8,9,10,11],"cluster_elements":[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.23119199999999995,"shell_mass_g":0.035568,"independent_dof":42,"update":{"candidates":[3,4,5],"accepted":[3,4,5],"thawed":[12,13,14,15],"new_independent":[9,10,11]}},{"step":2,"n":3,"free":[0,1,2,3,4,5,6,7,8],"rigid":[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[9,10,11,12,13,14],"cluster_elements":[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.22229999999999997,"shell_mass_g":0.044460000000000006,"independent_dof":51,"update":{"candidates":[6,7,8],"accepted":[6,7,8],"thawed":[16,17,18,19],"new_independent":[12,13,14]}},{"step":3,"n":3,"free":[0,1,2,3,4,5,6,7,8,9,10,11],"rigid":[18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[12,13,14,15,16,17],"cluster_elements":[24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.213408,"shell_mass_g":0.053352000000000004,"independent_dof":60,"update":{"candidates":[9,10,11],"accepted":[9,10,11],"thawed":[20,21,22,23],"new_independent":[15,16,17]}},{"step":4,"n":3,"free":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],"rigid":[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[15,16,17,18,19,20],"cluster_elements":[28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.20451599999999998,"shell_mass_g":0.062243999999999994,"independent_dof":69,"update":{"candidates":[12,13,14],"accepted":[12,13,14],"thawed":[24,25,26,27],"new_independent":[18,19,20]}},{"step":5,"n":3,"free":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],"rigid":[24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[18,19,20,21,22,23],"cluster_elements":[32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.195624,"shell_mass_g":0.071136,"independent_dof":78,"update":{"candidates":[15,16,17],"accepted":[15,16,17],"thawed":[28,29,30,31],"new_independent":[21,22,23]}},{"step":6,"n":3,"free":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"rigid":[27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[21,22,23,24,25,26],"cluster_elements":[36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.18673199999999995,"shell_mass_g":0.08002800000000002,"independent_dof":87,"update":{"candidates":[18,19,20],"accepted":[18,19,20],"thawed":[32,33,34,35],"new_independent":[24,25,26]}}]},"10":{"rows":31,"across":3,"rest_m":[[0.0,-0.0095,0.0],[0.0,0.0,0.0],[0.0,0.0095,0.0],[0.002,-0.0095,0.0],[0.002,0.0,0.0],[0.002,0.0095,0.0],[0.004,-0.0095,0.0],[0.004,0.0,0.0],[0.004,0.0095,0.0],[0.006,-0.0095,0.0],[0.006,0.0,0.0],[0.006,0.0095,0.0],[0.008,-0.0095,0.0],[0.008,0.0,0.0],[0.008,0.0095,0.0],[0.01,-0.0095,0.0],[0.01,0.0,0.0],[0.01,0.0095,0.0],[0.012,-0.0095,0.0],[0.012,0.0,0.0],[0.012,0.0095,0.0],[0.014,-0.0095,0.0],[0.014,0.0,0.0],[0.014,0.0095,0.0],[0.016,-0.0095,0.0],[0.016,0.0,0.0],[0.016,0.0095,0.0],[0.018000000000000002,-0.0095,0.0],[0.018000000000000002,0.0,0.0],[0.018000000000000002,0.0095,0.0],[0.02,-0.0095,0.0],[0.02,0.0,0.0],[0.02,0.0095,0.0],[0.022,-0.0095,0.0],[0.022,0.0,0.0],[0.022,0.0095,0.0],[0.024,-0.0095,0.0],[0.024,0.0,0.0],[0.024,0.0095,0.0],[0.026000000000000002,-0.0095,0.0],[0.026000000000000002,0.0,0.0],[0.026000000000000002,0.0095,0.0],[0.028,-0.0095,0.0],[0.028,0.0,0.0],[0.028,0.0095,0.0],[0.03,-0.0095,0.0],[0.03,0.0,0.0],[0.03,0.0095,0.0],[0.032,-0.0095,0.0],[0.032,0.0,0.0],[0.032,0.0095,0.0],[0.034,-0.0095,0.0],[0.034,0.0,0.0],[0.034,0.0095,0.0],[0.036000000000000004,-0.0095,0.0],[0.036000000000000004,0.0,0.0],[0.036000000000000004,0.0095,0.0],[0.038,-0.0095,0.0],[0.038,0.0,0.0],[0.038,0.0095,0.0],[0.04,-0.0095,0.0],[0.04,0.0,0.0],[0.04,0.0095,0.0],[0.042,-0.0095,0.0],[0.042,0.0,0.0],[0.042,0.0095,0.0],[0.044,-0.0095,0.0],[0.044,0.0,0.0],[0.044,0.0095,0.0],[0.046,-0.0095,0.0],[0.046,0.0,0.0],[0.046,0.0095,0.0],[0.048,-0.0095,0.0],[0.048,0.0,0.0],[0.048,0.0095,0.0],[0.05,-0.0095,0.0],[0.05,0.0,0.0],[0.05,0.0095,0.0],[0.052000000000000005,-0.0095,0.0],[0.052000000000000005,0.0,0.0],[0.052000000000000005,0.0095,0.0],[0.054,-0.0095,0.0],[0.054,0.0,0.0],[0.054,0.0095,0.0],[0.056,-0.0095,0.0],[0.056,0.0,0.0],[0.056,0.0095,0.0],[0.058,-0.0095,0.0],[0.058,0.0,0.0],[0.058,0.0095,0.0],[0.06,-0.0095,0.0],[0.06,0.0,0.0],[0.06,0.0095,0.0]],"triangles":[[0,3,1],[1,3,4],[1,4,2],[2,4,5],[3,6,4],[4,6,7],[4,7,5],[5,7,8],[6,9,7],[7,9,10],[7,10,8],[8,10,11],[9,12,10],[10,12,13],[10,13,11],[11,13,14],[12,15,13],[13,15,16],[13,16,14],[14,16,17],[15,18,16],[16,18,19],[16,19,17],[17,19,20],[18,21,19],[19,21,22],[19,22,20],[20,22,23],[21,24,22],[22,24,25],[22,25,23],[23,25,26],[24,27,25],[25,27,28],[25,28,26],[26,28,29],[27,30,28],[28,30,31],[28,31,29],[29,31,32],[30,33,31],[31,33,34],[31,34,32],[32,34,35],[33,36,34],[34,36,37],[34,37,35],[35,37,38],[36,39,37],[37,39,40],[37,40,38],[38,40,41],[39,42,40],[40,42,43],[40,43,41],[41,43,44],[42,45,43],[43,45,46],[43,46,44],[44,46,47],[45,48,46],[46,48,49],[46,49,47],[47,49,50],[48,51,49],[49,51,52],[49,52,50],[50,52,53],[51,54,52],[52,54,55],[52,55,53],[53,55,56],[54,57,55],[55,57,58],[55,58,56],[56,58,59],[57,60,58],[58,60,61],[58,61,59],[59,61,62],[60,63,61],[61,63,64],[61,64,62],[62,64,65],[63,66,64],[64,66,67],[64,67,65],[65,67,68],[66,69,67],[67,69,70],[67,70,68],[68,70,71],[69,72,70],[70,72,73],[70,73,71],[71,73,74],[72,75,73],[73,75,76],[73,76,74],[74,76,77],[75,78,76],[76,78,79],[76,79,77],[77,79,80],[78,81,79],[79,81,82],[79,82,80],[80,82,83],[81,84,82],[82,84,85],[82,85,83],[83,85,86],[84,87,85],[85,87,88],[85,88,86],[86,88,89],[87,90,88],[88,90,91],[88,91,89],[89,91,92]],"states":[{"step":0,"n":10,"free":[0,1,2],"rigid":[30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],"cluster_elements":[40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.17783999999999997,"shell_mass_g":0.08892000000000001,"independent_dof":96,"update":null},{"step":1,"n":10,"free":[0,1,2,3,4,5],"rigid":[33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],"cluster_elements":[44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.168948,"shell_mass_g":0.097812,"independent_dof":105,"update":{"candidates":[3,4,5],"accepted":[3,4,5],"thawed":[40,41,42,43],"new_independent":[30,31,32]}},{"step":2,"n":10,"free":[0,1,2,3,4,5,6,7,8],"rigid":[36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],"cluster_elements":[48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.16005599999999995,"shell_mass_g":0.106704,"independent_dof":114,"update":{"candidates":[6,7,8],"accepted":[6,7,8],"thawed":[44,45,46,47],"new_independent":[33,34,35]}},{"step":3,"n":10,"free":[0,1,2,3,4,5,6,7,8,9,10,11],"rigid":[39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38],"cluster_elements":[52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.15116399999999997,"shell_mass_g":0.115596,"independent_dof":123,"update":{"candidates":[9,10,11],"accepted":[9,10,11],"thawed":[48,49,50,51],"new_independent":[36,37,38]}},{"step":4,"n":10,"free":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],"rigid":[42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41],"cluster_elements":[56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.14227199999999998,"shell_mass_g":0.12448800000000002,"independent_dof":132,"update":{"candidates":[12,13,14],"accepted":[12,13,14],"thawed":[52,53,54,55],"new_independent":[39,40,41]}},{"step":5,"n":10,"free":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],"rigid":[45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44],"cluster_elements":[60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.13338,"shell_mass_g":0.13338,"independent_dof":141,"update":{"candidates":[15,16,17],"accepted":[15,16,17],"thawed":[56,57,58,59],"new_independent":[42,43,44]}},{"step":6,"n":10,"free":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"rigid":[48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92],"collar":[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],"cluster_elements":[64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119],"cluster_mass_g":0.12448799999999996,"shell_mass_g":0.14227200000000004,"independent_dof":150,"update":{"candidates":[18,19,20],"accepted":[18,19,20],"thawed":[60,61,62,63],"new_independent":[45,46,47]}}]}},"recursion_counterexample":{"correct_released":3,"wrong_released":6}}};
function bondScalar(lambda){
 const m=DEEP.math, factor=m.reference_volume_m3*m.stiffness_pa;
 return {energy:factor*(lambda*lambda-1)**2*1e6,
         force:-4*m.stiffness_pa*m.triangle_area_m2/3*lambda*(lambda*lambda-1),
         curvature:4*factor*(3*lambda*lambda-1)*1e6};
}
function drawBondMath(){
 const axis=$('math-axis').value, lambda=Number($('math-stretch').value)/100;
 const samples=Array.from({length:101},(_,i)=>({x:.5+i/100,y:bondScalar(.5+i/100)[axis]}));
 const values=samples.map(p=>p.y), low=Math.min(0,...values), high=Math.max(0,...values), spread=high-low;
 const yMin=low-spread*.08,yMax=high+spread*.08;
 const sx=x=>55+(x-.5)*535, sy=y=>235-(y-yMin)/(yMax-yMin)*210;
 const unit=axis==='force'?'N':'μJ', value=bondScalar(lambda), current=value[axis];
 const points=samples.map(p=>`${sx(p.x)},${sy(p.y)}`).join(' ');
 $('math-curve').innerHTML=`<line x1="55" y1="${sy(0)}" x2="590" y2="${sy(0)}" stroke="#bccbc4"/><line x1="55" y1="25" x2="55" y2="235" stroke="#64716d"/><polyline points="${points}" fill="none" stroke="#146b91" stroke-width="3"/><line x1="${sx(lambda)}" y1="25" x2="${sx(lambda)}" y2="235" stroke="#ad4e86" stroke-dasharray="4 4"/><circle cx="${sx(lambda)}" cy="${sy(current)}" r="5" fill="#ad4e86"/><text x="60" y="18" font-size="13">${axis==='energy'?'物理势能':axis==='force'?'P 的 z 向恢复力':'对 λ 的标量曲率'} / ${unit}</text><text x="4" y="${sy(high)+4}" font-size="12">${high.toFixed(1)}</text><text x="4" y="${sy(low)+4}" font-size="12">${low.toFixed(1)}</text><text x="46" y="260" font-size="12">0.5</text><text x="315" y="260" font-size="12">1.0</text><text x="578" y="260" font-size="12">1.5</text><text x="325" y="282" font-size="12">高度比 λ</text>`;
 $('math-stretch-value').textContent=lambda.toFixed(2);
 const force=Math.abs(value.force), release=lambda>1&&force>DEEP.math.break_force_n;
 const state=lambda<=1?'压缩或参考态，不触发这条拉伸断裂。':release?'张开且超过 0.5 N，步末释放。':'张开，但尚未超过 0.5 N。';
 $('math-sample').textContent=`λ=${lambda.toFixed(2)}；力模 ${force.toFixed(3)} N；E=${value.energy.toFixed(2)} μJ。${state}${value.curvature<0?' 此处精确标量曲率为负。':''}`;
 $('math-checks').textContent=`18 项二阶数学检查通过；Hessian 与梯度中心差分的最大归一化误差约 ${DEEP.math.hessian_error.toExponential(2)}。本教学几何的临界张开比约 ${DEEP.math.critical_normal_stretch.toFixed(8)}。`;
}
$('math-stretch').addEventListener('input',drawBondMath);$('math-axis').addEventListener('change',drawBondMath);drawBondMath();
function drawMesh(){
 const dataset=DEEP.mesh.datasets[$('mesh-n').value], step=Number($('mesh-step').value), state=dataset.states[step];
 const free=new Set(state.free), rigid=new Set(state.rigid), elements=new Set(state.cluster_elements);
 const coords=dataset.rest_m.map(v=>[26+v[0]/.06*948,90+(v[1]+.0095)/.019*105]);
 const polygons=dataset.triangles.map((t,i)=>`<polygon points="${t.map(v=>coords[v].join(',')).join(' ')}" fill="${elements.has(i)?'#f0d9a6':'#e6ecea'}" stroke="#c3cfc7" stroke-width=".8"/>`).join('');
 const nodes=coords.map(([x,y],i)=>`<circle cx="${x}" cy="${y}" r="3.6" fill="${free.has(i)?'#146b91':rigid.has(i)?'#c58a1a':'#ad4e86'}"/>`).join('');
 $('mesh-diagram').innerHTML=`<text x="26" y="31" font-size="18">平坦材料坐标 · 31 行 × 3 顶点 / 120 三角形</text><text x="26" y="60" font-size="14">元素填色表示归属；顶点颜色表示释放状态与运动坐标</text>${polygons}${nodes}<text x="26" y="228" font-size="13">自由端</text><text x="910" y="228" font-size="13">卷内方向</text>`;
 $('mesh-step-value').textContent=String(step);
 $('mesh-stats').innerHTML=`<div><div class="number">${state.cluster_elements.length}</div><div class="caption">cluster 三角形 / 胶带运动 DOF ${state.independent_dof}</div></div><div><div class="number">${(state.cluster_mass_g+state.shell_mass_g).toFixed(6)} g</div><div class="caption">cluster ${state.cluster_mass_g.toFixed(6)} + shell ${state.shell_mass_g.toFixed(6)}</div></div>`;
 $('mesh-explanation').textContent=step===0?`N=${state.n} 的初始分类。距离恰为 N 的 cluster 元素不恢复；只有某个角点距离严格小于 N 才恢复。`:`第 ${step} 次更新：相对旧自由区接受 ${state.update.accepted.length} 个顶点；恢复 ${state.update.thawed.length} 个三角形，${state.update.new_independent.length} 个顶点新获得独立坐标。更换质量中心后保持同一刚体速度场。`;
 $('mesh-checks').textContent=`80 项网格与状态检查通过；守恒检查覆盖重分类瞬间。输入位移由程序指定，没有薄壳/IPC 动力学积分。原地递归的错误更新在 N=3 示例中一次接受 ${DEEP.mesh.recursion_counterexample.wrong_released} 个顶点，正确版本为 ${DEEP.mesh.recursion_counterexample.correct_released} 个。`;
}
$('mesh-n').addEventListener('change',drawMesh);$('mesh-step').addEventListener('input',drawMesh);drawMesh();
