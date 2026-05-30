{
  "_metadata": {
    "organ": "gi",
    "schema_version": "1.0",
    "description": "Gastrointestinal state model. Used by gastroenteritis, GERD, IBD, cholecystitis, appendicitis (when added).",
    "source": "Boron & Boulpaep GI Physiology; Harrison's IM Gastroenterology section",
    "review_status": "unreviewed"
  },
  "state_variables": {
    "mucosal_inflammation": {
      "range": [
        0,
        1
      ],
      "description": "GI mucosal/submucosal inflammation"
    },
    "motility": {
      "range": [
        0,
        1
      ],
      "description": "peristalsis intensity — normal=0.5, hypermotility=high, ileus=low"
    },
    "secretion": {
      "range": [
        0,
        1
      ],
      "description": "luminal fluid secretion (cAMP-driven)"
    },
    "acid_exposure": {
      "range": [
        0,
        1
      ],
      "description": "esophageal/gastric mucosa exposure to acid"
    },
    "lower_esophageal_sphincter_tone": {
      "range": [
        0,
        1
      ],
      "description": "LES competence — HIGHER = better closure",
      "inverse": true
    },
    "visceral_nociception": {
      "range": [
        0,
        1
      ],
      "description": "afferent pain signaling from GI wall"
    },
    "vomiting_center_activation": {
      "range": [
        0,
        1
      ],
      "description": "medullary CTZ + vomiting center drive"
    },
    "luminal_pathogen": {
      "range": [
        0,
        1
      ],
      "description": "enteric pathogen load"
    },
    "absorption_capacity": {
      "range": [
        0,
        1
      ],
      "description": "fluid/nutrient absorption — HIGHER = better",
      "inverse": true
    }
  },
  "derivation_rules": [
    {
      "symptom": "diarrhea",
      "condition": {
        "any": [
          {
            "all": [
              {
                "state": "secretion",
                "op": ">",
                "threshold": 0.5
              },
              {
                "state": "motility",
                "op": ">",
                "threshold": 0.5
              }
            ]
          },
          {
            "state": "absorption_capacity",
            "op": "<",
            "threshold": 0.4
          }
        ]
      },
      "rationale": "increased intestinal secretion + hypermotility OR impaired absorption → loose stools",
      "confidence": 1.0
    },
    {
      "symptom": "nausea",
      "condition": {
        "state": "vomiting_center_activation",
        "op": ">",
        "threshold": 0.4
      },
      "rationale": "medullary CTZ activation from pathogen toxins, distension, or visceral afferents → nausea sensation",
      "confidence": 1.0
    },
    {
      "symptom": "vomiting",
      "condition": {
        "state": "vomiting_center_activation",
        "op": ">",
        "threshold": 0.6
      },
      "rationale": "strong vomiting center drive triggers reverse peristalsis + diaphragm contraction",
      "confidence": 1.0
    },
    {
      "symptom": "abdominal cramps",
      "condition": {
        "all": [
          {
            "state": "motility",
            "op": ">",
            "threshold": 0.5
          },
          {
            "state": "visceral_nociception",
            "op": ">",
            "threshold": 0.4
          }
        ]
      },
      "rationale": "hypermotility + visceral afferent activation → crampy intermittent pain",
      "confidence": 1.0
    },
    {
      "symptom": "abdominal pain",
      "condition": {
        "state": "visceral_nociception",
        "op": ">",
        "threshold": 0.5
      },
      "rationale": "visceral afferent nociceptors activated by inflammation or distension",
      "confidence": 0.9
    },
    {
      "symptom": "heartburn",
      "condition": {
        "state": "acid_exposure",
        "op": ">",
        "threshold": 0.5
      },
      "rationale": "acid contact with esophageal mucosa activates afferent fibers — burning retrosternal sensation",
      "confidence": 1.0
    },
    {
      "symptom": "acid reflux",
      "condition": {
        "all": [
          {
            "state": "lower_esophageal_sphincter_tone",
            "op": "<",
            "threshold": 0.5
          },
          {
            "state": "acid_exposure",
            "op": ">",
            "threshold": 0.4
          }
        ]
      },
      "rationale": "LES incompetence allows gastric content to reflux into esophagus",
      "confidence": 1.0
    },
    {
      "symptom": "regurgitation",
      "condition": {
        "state": "lower_esophageal_sphincter_tone",
        "op": "<",
        "threshold": 0.3
      },
      "rationale": "severe LES dysfunction allows passive reflux of food/fluid to mouth",
      "confidence": 0.9
    },
    {
      "symptom": "fever",
      "condition": {
        "state": "mucosal_inflammation",
        "op": ">",
        "threshold": 0.6
      },
      "rationale": "GI inflammation releases pyrogens → hypothalamic setpoint elevation",
      "confidence": 0.8
    },
    {
      "symptom": "loss of appetite",
      "condition": {
        "state": "mucosal_inflammation",
        "op": ">",
        "threshold": 0.5
      },
      "rationale": "inflammatory cytokines suppress hypothalamic appetite centers",
      "confidence": 0.8
    }
  ],
  "diseases": {
    "Acute Gastroenteritis": {
      "review_status": "unreviewed",
      "description": "Viral (norovirus, rotavirus) or bacterial enteritis. Pathogen triggers secretion + hypermotility + vomiting → diarrhea/nausea/cramps.",
      "perturbations": [
        {
          "variable": "luminal_pathogen",
          "delta": 0.8,
          "cause": "enteric pathogen colonization"
        },
        {
          "variable": "mucosal_inflammation",
          "delta": 0.65,
          "cause": "epithelial damage from pathogen + cytokine response"
        },
        {
          "variable": "secretion",
          "delta": 0.7,
          "cause": "toxin-stimulated chloride/water secretion"
        },
        {
          "variable": "motility",
          "delta": 0.65,
          "cause": "irritated bowel hypermotility (host defense to expel pathogen)"
        },
        {
          "variable": "visceral_nociception",
          "delta": 0.55,
          "cause": "inflammation activates mesenteric afferents"
        },
        {
          "variable": "vomiting_center_activation",
          "delta": 0.6,
          "cause": "toxins reach CTZ via bloodstream + vagal afferents"
        },
        {
          "variable": "absorption_capacity",
          "delta": -0.5,
          "cause": "damaged enterocytes lose absorptive function"
        }
      ]
    },
    "Gastroesophageal Reflux Disease": {
      "review_status": "unreviewed",
      "description": "Gastroesophageal reflux disease — LES incompetence allows gastric acid to reflux into esophagus, causing mucosal irritation.",
      "perturbations": [
        {
          "variable": "lower_esophageal_sphincter_tone",
          "delta": -0.55,
          "cause": "transient LES relaxations OR reduced resting tone"
        },
        {
          "variable": "acid_exposure",
          "delta": 0.65,
          "cause": "gastric acid contacts esophageal mucosa repeatedly"
        },
        {
          "variable": "mucosal_inflammation",
          "delta": 0.4,
          "cause": "esophagitis from chronic acid exposure"
        },
        {
          "variable": "visceral_nociception",
          "delta": 0.4,
          "cause": "afferent activation from acid-irritated esophagus"
        }
      ]
    }
  }
}