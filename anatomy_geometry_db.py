"""
Anatomy Geometry Database
═══════════════════════════════════════════════════════════════
Per-organ 3D bounding box + volume for ALL 208 organs.

Data sources:
  • Visible Human Project (NLM) — organ dimensions
  • TA98 / Foundational Model of Anatomy
  • Standard surgical anatomy references
  • Gray's Anatomy 41st edition organ measurements

Coordinate system (matches anatomy_atlas.py):
  x = lateral    (negative=left, positive=right)
  y = vertical   (positive=up, negative=down)
  z = front-back (positive=anterior, negative=posterior)

Units: normalized body units (1.0 = ~head-to-toe ≈ 170cm adult)
       extent values are HALF-extents (so 0.04 = 8cm full width)

Volume: in mL (cubic centimeters)
"""

# ═══════════════════════════════════════════════════════════════
# ORGAN GEOMETRY — comprehensive coverage of all 208 organs
# ═══════════════════════════════════════════════════════════════
# Format: name → {extent: (dx, dy, dz), volume_ml: V, surgical_zone: str}

_FALLBACK_ORGAN_GEOMETRY = {

    # ════════════════════════════════════════════
    # NEUROLOGIC SYSTEM
    # ════════════════════════════════════════════
    "brain":              {"extent": (0.075, 0.060, 0.070), "volume_ml": 1400,
                           "surgical_zone": "intracranial"},
    "cerebellum":         {"extent": (0.060, 0.030, 0.040), "volume_ml": 150,
                           "surgical_zone": "posterior_fossa"},
    "brainstem":          {"extent": (0.015, 0.040, 0.025), "volume_ml": 30,
                           "surgical_zone": "posterior_fossa"},
    "spinal_cord":        {"extent": (0.012, 0.300, 0.015), "volume_ml": 35,
                           "surgical_zone": "spinal_canal"},
    "meninges":           {"extent": (0.085, 0.070, 0.080), "volume_ml": 50,
                           "surgical_zone": "intracranial"},
    "vagus_n":            {"extent": (0.010, 0.250, 0.020), "volume_ml": 5,
                           "surgical_zone": "neck_thorax"},
    "phrenic_n":          {"extent": (0.008, 0.150, 0.020), "volume_ml": 3,
                           "surgical_zone": "neck_thorax"},

    # ════════════════════════════════════════════
    # CARDIOVASCULAR SYSTEM
    # ════════════════════════════════════════════
    # Heart chambers
    "heart":              {"extent": (0.060, 0.060, 0.050), "volume_ml": 300,
                           "surgical_zone": "mediastinum"},
    "right_atrium":       {"extent": (0.030, 0.030, 0.025), "volume_ml": 60,
                           "surgical_zone": "mediastinum"},
    "right_ventricle":    {"extent": (0.040, 0.050, 0.040), "volume_ml": 80,
                           "surgical_zone": "mediastinum"},
    "left_atrium":        {"extent": (0.030, 0.030, 0.020), "volume_ml": 60,
                           "surgical_zone": "mediastinum"},
    "left_ventricle":     {"extent": (0.040, 0.050, 0.040), "volume_ml": 130,
                           "surgical_zone": "mediastinum"},

    # Great vessels
    "aorta":              {"extent": (0.012, 0.020, 0.012), "volume_ml": 60,
                           "surgical_zone": "mediastinum"},
    "aortic_arch":        {"extent": (0.040, 0.020, 0.020), "volume_ml": 30,
                           "surgical_zone": "superior_mediastinum"},
    "thoracic_aorta":     {"extent": (0.012, 0.150, 0.012), "volume_ml": 80,
                           "surgical_zone": "posterior_mediastinum"},
    "abdominal_aorta":    {"extent": (0.012, 0.150, 0.012), "volume_ml": 70,
                           "surgical_zone": "retroperitoneum"},
    "ivc":                {"extent": (0.015, 0.250, 0.015), "volume_ml": 150,
                           "surgical_zone": "retroperitoneum"},
    "svc":                {"extent": (0.012, 0.060, 0.012), "volume_ml": 40,
                           "surgical_zone": "superior_mediastinum"},
    "pulm_aa":            {"extent": (0.015, 0.020, 0.015), "volume_ml": 30,
                           "surgical_zone": "mediastinum"},
    "pulm_vv":            {"extent": (0.015, 0.020, 0.015), "volume_ml": 30,
                           "surgical_zone": "mediastinum"},
    "coronary_aa":        {"extent": (0.020, 0.020, 0.015), "volume_ml": 5,
                           "surgical_zone": "mediastinum"},
    "coronary_sinus":     {"extent": (0.025, 0.010, 0.015), "volume_ml": 8,
                           "surgical_zone": "mediastinum"},

    # Coronary detail (newly added)
    "left_main_coronary": {"extent": (0.005, 0.005, 0.005), "volume_ml": 1,
                           "surgical_zone": "epicardium"},
    "lad_a":              {"extent": (0.003, 0.040, 0.003), "volume_ml": 2,
                           "surgical_zone": "epicardium"},
    "lcx_a":              {"extent": (0.030, 0.005, 0.005), "volume_ml": 2,
                           "surgical_zone": "epicardium"},
    "rca_a":              {"extent": (0.030, 0.005, 0.005), "volume_ml": 2,
                           "surgical_zone": "epicardium"},

    # Cerebral arteries
    "internal_carotid_a":   {"extent": (0.005, 0.080, 0.005), "volume_ml": 5,
                             "surgical_zone": "neck_skull_base"},
    "l_internal_carotid_a": {"extent": (0.005, 0.080, 0.005), "volume_ml": 5,
                             "surgical_zone": "neck_skull_base"},
    "external_carotid_a":   {"extent": (0.005, 0.060, 0.008), "volume_ml": 4,
                             "surgical_zone": "neck"},
    "l_external_carotid_a": {"extent": (0.005, 0.060, 0.008), "volume_ml": 4,
                             "surgical_zone": "neck"},
    "anterior_cerebral_a":  {"extent": (0.020, 0.005, 0.010), "volume_ml": 2,
                             "surgical_zone": "intracranial"},
    "middle_cerebral_a":    {"extent": (0.030, 0.005, 0.005), "volume_ml": 3,
                             "surgical_zone": "intracranial"},
    "l_middle_cerebral_a":  {"extent": (0.030, 0.005, 0.005), "volume_ml": 3,
                             "surgical_zone": "intracranial"},
    "posterior_cerebral_a": {"extent": (0.025, 0.005, 0.020), "volume_ml": 2,
                             "surgical_zone": "intracranial"},
    "anterior_communicating_a": {"extent": (0.005, 0.003, 0.003), "volume_ml": 0.5,
                                 "surgical_zone": "intracranial"},
    "posterior_communicating_a": {"extent": (0.010, 0.003, 0.003), "volume_ml": 0.5,
                                  "surgical_zone": "intracranial"},
    "ophthalmic_a":         {"extent": (0.010, 0.005, 0.020), "volume_ml": 1,
                             "surgical_zone": "orbit"},
    "basilar_a":            {"extent": (0.005, 0.020, 0.008), "volume_ml": 2,
                             "surgical_zone": "intracranial"},
    "r_vertebral_a":        {"extent": (0.005, 0.060, 0.008), "volume_ml": 3,
                             "surgical_zone": "neck"},
    "l_vertebral_a":        {"extent": (0.005, 0.060, 0.008), "volume_ml": 3,
                             "surgical_zone": "neck"},
    "r_carotid_a":          {"extent": (0.005, 0.080, 0.008), "volume_ml": 5,
                             "surgical_zone": "neck"},
    "l_carotid_a":          {"extent": (0.005, 0.080, 0.008), "volume_ml": 5,
                             "surgical_zone": "neck"},

    # Cerebral venous sinuses
    "superior_sagittal_sinus": {"extent": (0.015, 0.080, 0.015), "volume_ml": 8,
                                "surgical_zone": "intracranial"},
    "transverse_sinus":     {"extent": (0.030, 0.005, 0.015), "volume_ml": 5,
                             "surgical_zone": "intracranial"},
    "sigmoid_sinus":        {"extent": (0.005, 0.030, 0.020), "volume_ml": 4,
                             "surgical_zone": "intracranial"},
    "cavernous_sinus":      {"extent": (0.020, 0.015, 0.015), "volume_ml": 3,
                             "surgical_zone": "skull_base"},

    # Pulmonary detail
    "r_pulm_a":             {"extent": (0.025, 0.008, 0.015), "volume_ml": 15,
                             "surgical_zone": "mediastinum"},
    "l_pulm_a":             {"extent": (0.025, 0.008, 0.015), "volume_ml": 15,
                             "surgical_zone": "mediastinum"},
    "r_upper_pulm_v":       {"extent": (0.020, 0.005, 0.010), "volume_ml": 8,
                             "surgical_zone": "mediastinum"},
    "r_lower_pulm_v":       {"extent": (0.020, 0.005, 0.010), "volume_ml": 8,
                             "surgical_zone": "mediastinum"},
    "l_upper_pulm_v":       {"extent": (0.020, 0.005, 0.010), "volume_ml": 8,
                             "surgical_zone": "mediastinum"},
    "l_lower_pulm_v":       {"extent": (0.020, 0.005, 0.010), "volume_ml": 8,
                             "surgical_zone": "mediastinum"},
    "bronchial_a":          {"extent": (0.015, 0.030, 0.010), "volume_ml": 2,
                             "surgical_zone": "mediastinum"},

    # Subclavian / brachiocephalic
    "brachiocephalic_a":    {"extent": (0.015, 0.020, 0.010), "volume_ml": 5,
                             "surgical_zone": "superior_mediastinum"},
    "r_subclavian_a":       {"extent": (0.025, 0.008, 0.010), "volume_ml": 8,
                             "surgical_zone": "thoracic_inlet"},
    "l_subclavian_a":       {"extent": (0.025, 0.008, 0.010), "volume_ml": 8,
                             "surgical_zone": "thoracic_inlet"},
    "r_subclavian_v":       {"extent": (0.025, 0.008, 0.010), "volume_ml": 10,
                             "surgical_zone": "thoracic_inlet"},
    "l_subclavian_v":       {"extent": (0.025, 0.008, 0.010), "volume_ml": 10,
                             "surgical_zone": "thoracic_inlet"},
    "r_brachiocephalic_v":  {"extent": (0.025, 0.010, 0.010), "volume_ml": 8,
                             "surgical_zone": "superior_mediastinum"},
    "l_brachiocephalic_v":  {"extent": (0.040, 0.010, 0.010), "volume_ml": 12,
                             "surgical_zone": "superior_mediastinum"},
    "azygos_v":             {"extent": (0.005, 0.150, 0.008), "volume_ml": 8,
                             "surgical_zone": "posterior_mediastinum"},

    # Jugular
    "r_jugular_v":          {"extent": (0.010, 0.060, 0.012), "volume_ml": 12,
                             "surgical_zone": "neck"},
    "l_jugular_v":          {"extent": (0.010, 0.060, 0.012), "volume_ml": 12,
                             "surgical_zone": "neck"},

    # Upper limb arterial
    "axillary_a":           {"extent": (0.015, 0.030, 0.010), "volume_ml": 6,
                             "surgical_zone": "axilla"},
    "l_axillary_a":         {"extent": (0.015, 0.030, 0.010), "volume_ml": 6,
                             "surgical_zone": "axilla"},
    "r_brachial_a":         {"extent": (0.005, 0.080, 0.008), "volume_ml": 5,
                             "surgical_zone": "upper_arm"},
    "l_brachial_a":         {"extent": (0.005, 0.080, 0.008), "volume_ml": 5,
                             "surgical_zone": "upper_arm"},
    "deep_brachial_a":      {"extent": (0.005, 0.060, 0.008), "volume_ml": 3,
                             "surgical_zone": "upper_arm"},
    "r_radial_a":           {"extent": (0.003, 0.080, 0.005), "volume_ml": 2,
                             "surgical_zone": "forearm"},
    "l_radial_a":           {"extent": (0.003, 0.080, 0.005), "volume_ml": 2,
                             "surgical_zone": "forearm"},
    "ulnar_a":              {"extent": (0.003, 0.080, 0.005), "volume_ml": 2,
                             "surgical_zone": "forearm"},
    "l_ulnar_a":            {"extent": (0.003, 0.080, 0.005), "volume_ml": 2,
                             "surgical_zone": "forearm"},
    "palmar_arch_aa":       {"extent": (0.020, 0.005, 0.005), "volume_ml": 1,
                             "surgical_zone": "hand"},
    "l_palmar_arch_aa":     {"extent": (0.020, 0.005, 0.005), "volume_ml": 1,
                             "surgical_zone": "hand"},

    # Upper limb venous
    "axillary_v":           {"extent": (0.015, 0.030, 0.010), "volume_ml": 8,
                             "surgical_zone": "axilla"},
    "l_axillary_v":         {"extent": (0.015, 0.030, 0.010), "volume_ml": 8,
                             "surgical_zone": "axilla"},
    "cephalic_v":           {"extent": (0.005, 0.150, 0.008), "volume_ml": 6,
                             "surgical_zone": "upper_limb"},
    "basilic_v":            {"extent": (0.005, 0.150, 0.008), "volume_ml": 6,
                             "surgical_zone": "upper_limb"},
    "median_cubital_v":     {"extent": (0.020, 0.005, 0.005), "volume_ml": 2,
                             "surgical_zone": "antecubital"},

    # Abdominal arteries
    "celiac":               {"extent": (0.020, 0.005, 0.010), "volume_ml": 4,
                             "surgical_zone": "retroperitoneum"},
    "sma":                  {"extent": (0.005, 0.040, 0.010), "volume_ml": 6,
                             "surgical_zone": "retroperitoneum"},
    "ima":                  {"extent": (0.005, 0.030, 0.010), "volume_ml": 4,
                             "surgical_zone": "retroperitoneum"},
    "splenic_a":            {"extent": (0.040, 0.005, 0.010), "volume_ml": 5,
                             "surgical_zone": "retroperitoneum"},
    "common_hepatic_a":     {"extent": (0.020, 0.005, 0.010), "volume_ml": 3,
                             "surgical_zone": "retroperitoneum"},
    "proper_hepatic_a":     {"extent": (0.010, 0.020, 0.005), "volume_ml": 2,
                             "surgical_zone": "porta_hepatis"},
    "hepatic_a":            {"extent": (0.020, 0.005, 0.010), "volume_ml": 3,
                             "surgical_zone": "porta_hepatis"},
    "left_gastric_a":       {"extent": (0.020, 0.005, 0.005), "volume_ml": 2,
                             "surgical_zone": "lesser_omentum"},
    "right_gastric_a":      {"extent": (0.020, 0.005, 0.005), "volume_ml": 2,
                             "surgical_zone": "lesser_omentum"},
    "gastroduodenal_a":     {"extent": (0.005, 0.020, 0.005), "volume_ml": 2,
                             "surgical_zone": "duodenum"},
    "cystic_a":             {"extent": (0.005, 0.010, 0.005), "volume_ml": 1,
                             "surgical_zone": "porta_hepatis"},
    "ileocolic_a":          {"extent": (0.020, 0.030, 0.010), "volume_ml": 2,
                             "surgical_zone": "rlq"},
    "right_colic_a":        {"extent": (0.020, 0.005, 0.010), "volume_ml": 2,
                             "surgical_zone": "right_abdomen"},
    "middle_colic_a":       {"extent": (0.030, 0.005, 0.010), "volume_ml": 2,
                             "surgical_zone": "transverse_meso"},
    "left_colic_a":         {"extent": (0.020, 0.005, 0.010), "volume_ml": 2,
                             "surgical_zone": "left_abdomen"},
    "sigmoid_aa":           {"extent": (0.020, 0.020, 0.010), "volume_ml": 2,
                             "surgical_zone": "sigmoid_meso"},
    "superior_rectal_a":    {"extent": (0.005, 0.020, 0.005), "volume_ml": 1,
                             "surgical_zone": "pelvis"},

    # Renal vessels
    "r_renal_a":            {"extent": (0.025, 0.005, 0.010), "volume_ml": 3,
                             "surgical_zone": "retroperitoneum"},
    "l_renal_a":            {"extent": (0.025, 0.005, 0.010), "volume_ml": 3,
                             "surgical_zone": "retroperitoneum"},
    "r_renal_v":            {"extent": (0.025, 0.005, 0.010), "volume_ml": 4,
                             "surgical_zone": "retroperitoneum"},
    "l_renal_v":            {"extent": (0.040, 0.005, 0.010), "volume_ml": 5,
                             "surgical_zone": "retroperitoneum"},

    # Portal system
    "portal_vein":          {"extent": (0.005, 0.040, 0.010), "volume_ml": 8,
                             "surgical_zone": "porta_hepatis"},
    "smv":                  {"extent": (0.005, 0.040, 0.008), "volume_ml": 6,
                             "surgical_zone": "retroperitoneum"},
    "imv":                  {"extent": (0.005, 0.030, 0.008), "volume_ml": 4,
                             "surgical_zone": "retroperitoneum"},
    "splenic_v":            {"extent": (0.040, 0.005, 0.008), "volume_ml": 5,
                             "surgical_zone": "retroperitoneum"},
    "hepatic_vv":           {"extent": (0.020, 0.005, 0.010), "volume_ml": 4,
                             "surgical_zone": "liver"},
    "left_gastric_v":       {"extent": (0.015, 0.005, 0.005), "volume_ml": 1,
                             "surgical_zone": "lesser_omentum"},
    "paraumbilical_vv":     {"extent": (0.010, 0.030, 0.010), "volume_ml": 1,
                             "surgical_zone": "abdominal_wall"},
    "cystic_v":             {"extent": (0.005, 0.010, 0.005), "volume_ml": 1,
                             "surgical_zone": "porta_hepatis"},

    # Pelvic
    "r_iliac_a":            {"extent": (0.005, 0.030, 0.008), "volume_ml": 5,
                             "surgical_zone": "pelvis"},
    "l_iliac_a":            {"extent": (0.005, 0.030, 0.008), "volume_ml": 5,
                             "surgical_zone": "pelvis"},
    "internal_iliac_a":     {"extent": (0.008, 0.030, 0.010), "volume_ml": 4,
                             "surgical_zone": "pelvis"},
    "l_internal_iliac_a":   {"extent": (0.008, 0.030, 0.010), "volume_ml": 4,
                             "surgical_zone": "pelvis"},
    "external_iliac_a":     {"extent": (0.005, 0.030, 0.008), "volume_ml": 4,
                             "surgical_zone": "pelvis"},
    "l_external_iliac_a":   {"extent": (0.005, 0.030, 0.008), "volume_ml": 4,
                             "surgical_zone": "pelvis"},
    "uterine_a":            {"extent": (0.020, 0.020, 0.010), "volume_ml": 2,
                             "surgical_zone": "pelvis_female"},
    "ovarian_a":            {"extent": (0.005, 0.040, 0.010), "volume_ml": 2,
                             "surgical_zone": "pelvis_female"},
    "testicular_a":         {"extent": (0.005, 0.060, 0.010), "volume_ml": 2,
                             "surgical_zone": "scrotum"},
    "vesical_aa":           {"extent": (0.020, 0.010, 0.010), "volume_ml": 2,
                             "surgical_zone": "pelvis"},
    "obturator_a":          {"extent": (0.020, 0.010, 0.010), "volume_ml": 2,
                             "surgical_zone": "pelvis"},
    "r_iliac_v":            {"extent": (0.008, 0.030, 0.010), "volume_ml": 8,
                             "surgical_zone": "pelvis"},
    "l_iliac_v":            {"extent": (0.008, 0.030, 0.010), "volume_ml": 8,
                             "surgical_zone": "pelvis"},

    # Lower limb arterial
    "r_femoral_a":          {"extent": (0.005, 0.060, 0.008), "volume_ml": 8,
                             "surgical_zone": "femoral_triangle"},
    "l_femoral_a":          {"extent": (0.005, 0.060, 0.008), "volume_ml": 8,
                             "surgical_zone": "femoral_triangle"},
    "deep_femoral_a":       {"extent": (0.005, 0.080, 0.008), "volume_ml": 6,
                             "surgical_zone": "thigh_deep"},
    "l_deep_femoral_a":     {"extent": (0.005, 0.080, 0.008), "volume_ml": 6,
                             "surgical_zone": "thigh_deep"},
    "r_popliteal_a":        {"extent": (0.005, 0.030, 0.008), "volume_ml": 4,
                             "surgical_zone": "popliteal_fossa"},
    "l_popliteal_a":        {"extent": (0.005, 0.030, 0.008), "volume_ml": 4,
                             "surgical_zone": "popliteal_fossa"},
    "r_tibial_a":           {"extent": (0.005, 0.080, 0.005), "volume_ml": 3,
                             "surgical_zone": "calf"},
    "l_tibial_a":           {"extent": (0.005, 0.080, 0.005), "volume_ml": 3,
                             "surgical_zone": "calf"},
    "anterior_tibial_a":    {"extent": (0.003, 0.080, 0.005), "volume_ml": 2,
                             "surgical_zone": "calf_anterior"},
    "l_anterior_tibial_a":  {"extent": (0.003, 0.080, 0.005), "volume_ml": 2,
                             "surgical_zone": "calf_anterior"},
    "posterior_tibial_a":   {"extent": (0.003, 0.080, 0.005), "volume_ml": 2,
                             "surgical_zone": "calf_posterior"},
    "l_posterior_tibial_a": {"extent": (0.003, 0.080, 0.005), "volume_ml": 2,
                             "surgical_zone": "calf_posterior"},
    "peroneal_a":           {"extent": (0.003, 0.080, 0.005), "volume_ml": 1,
                             "surgical_zone": "calf_posterior"},
    "dorsalis_pedis_a":     {"extent": (0.020, 0.005, 0.005), "volume_ml": 1,
                             "surgical_zone": "foot_dorsum"},
    "l_dorsalis_pedis_a":   {"extent": (0.020, 0.005, 0.005), "volume_ml": 1,
                             "surgical_zone": "foot_dorsum"},

    # Lower limb venous
    "r_femoral_v":          {"extent": (0.005, 0.060, 0.008), "volume_ml": 12,
                             "surgical_zone": "femoral_triangle"},
    "l_femoral_v":          {"extent": (0.005, 0.060, 0.008), "volume_ml": 12,
                             "surgical_zone": "femoral_triangle"},
    "deep_femoral_v":       {"extent": (0.005, 0.080, 0.008), "volume_ml": 8,
                             "surgical_zone": "thigh_deep"},
    "r_popliteal_v":        {"extent": (0.005, 0.030, 0.008), "volume_ml": 6,
                             "surgical_zone": "popliteal_fossa"},
    "l_popliteal_v":        {"extent": (0.005, 0.030, 0.008), "volume_ml": 6,
                             "surgical_zone": "popliteal_fossa"},
    "r_saphenous_v":        {"extent": (0.005, 0.150, 0.005), "volume_ml": 8,
                             "surgical_zone": "lower_limb_superficial"},
    "l_saphenous_v":        {"extent": (0.005, 0.150, 0.005), "volume_ml": 8,
                             "surgical_zone": "lower_limb_superficial"},
    "great_saphenous_v":    {"extent": (0.005, 0.250, 0.005), "volume_ml": 12,
                             "surgical_zone": "lower_limb_superficial"},
    "l_great_saphenous_v":  {"extent": (0.005, 0.250, 0.005), "volume_ml": 12,
                             "surgical_zone": "lower_limb_superficial"},
    "small_saphenous_v":    {"extent": (0.005, 0.080, 0.005), "volume_ml": 4,
                             "surgical_zone": "calf_posterior"},
    "soleal_vv":            {"extent": (0.005, 0.080, 0.005), "volume_ml": 5,
                             "surgical_zone": "calf_deep"},
    "gastrocnemius_vv":     {"extent": (0.010, 0.040, 0.010), "volume_ml": 3,
                             "surgical_zone": "calf_posterior"},
    "perforator_vv":        {"extent": (0.005, 0.005, 0.005), "volume_ml": 1,
                             "surgical_zone": "lower_limb"},

    # ════════════════════════════════════════════
    # RESPIRATORY
    # ════════════════════════════════════════════
    "trachea":              {"extent": (0.012, 0.060, 0.015), "volume_ml": 30,
                             "surgical_zone": "anterior_neck"},
    "r_bronchus":           {"extent": (0.020, 0.020, 0.010), "volume_ml": 8,
                             "surgical_zone": "mediastinum"},
    "l_bronchus":           {"extent": (0.020, 0.020, 0.010), "volume_ml": 8,
                             "surgical_zone": "mediastinum"},
    "r_lung":               {"extent": (0.080, 0.100, 0.080), "volume_ml": 2700,
                             "surgical_zone": "right_pleural"},
    "l_lung":               {"extent": (0.080, 0.100, 0.080), "volume_ml": 2300,
                             "surgical_zone": "left_pleural"},
    "lungs":                {"extent": (0.180, 0.120, 0.080), "volume_ml": 5000,
                             "surgical_zone": "thorax"},
    "diaphragm":            {"extent": (0.150, 0.020, 0.100), "volume_ml": 200,
                             "surgical_zone": "thoracoabdominal"},
    "pleura":               {"extent": (0.180, 0.120, 0.080), "volume_ml": 50,
                             "surgical_zone": "pleural_space"},

    # ════════════════════════════════════════════
    # GASTROINTESTINAL
    # ════════════════════════════════════════════
    "mouth":                {"extent": (0.030, 0.030, 0.040), "volume_ml": 30,
                             "surgical_zone": "oral_cavity"},
    "pharynx":              {"extent": (0.025, 0.060, 0.030), "volume_ml": 50,
                             "surgical_zone": "pharynx"},
    "esophagus":            {"extent": (0.015, 0.150, 0.020), "volume_ml": 30,
                             "surgical_zone": "thorax_abdomen"},
    "stomach":              {"extent": (0.070, 0.050, 0.050), "volume_ml": 1000,
                             "surgical_zone": "left_upper_abdomen"},
    "duodenum":             {"extent": (0.050, 0.030, 0.040), "volume_ml": 30,
                             "surgical_zone": "retroperitoneum"},
    "jejunum":              {"extent": (0.100, 0.060, 0.060), "volume_ml": 800,
                             "surgical_zone": "left_central_abdomen"},
    "ileum":                {"extent": (0.100, 0.060, 0.060), "volume_ml": 800,
                             "surgical_zone": "right_central_abdomen"},
    "cecum":                {"extent": (0.050, 0.040, 0.040), "volume_ml": 200,
                             "surgical_zone": "rlq"},
    "appendix":             {"extent": (0.010, 0.020, 0.010), "volume_ml": 5,
                             "surgical_zone": "rlq"},
    "asc_colon":            {"extent": (0.040, 0.100, 0.040), "volume_ml": 250,
                             "surgical_zone": "right_abdomen"},
    "trans_colon":          {"extent": (0.150, 0.030, 0.040), "volume_ml": 250,
                             "surgical_zone": "central_abdomen"},
    "desc_colon":           {"extent": (0.040, 0.100, 0.040), "volume_ml": 250,
                             "surgical_zone": "left_abdomen"},
    "sig_colon":            {"extent": (0.050, 0.050, 0.040), "volume_ml": 150,
                             "surgical_zone": "llq_pelvis"},
    "rectum":               {"extent": (0.030, 0.050, 0.030), "volume_ml": 100,
                             "surgical_zone": "pelvis"},
    "anus":                 {"extent": (0.020, 0.020, 0.020), "volume_ml": 10,
                             "surgical_zone": "perineum"},

    # ════════════════════════════════════════════
    # HEPATOBILIARY
    # ════════════════════════════════════════════
    "liver":                {"extent": (0.100, 0.050, 0.060), "volume_ml": 1500,
                             "surgical_zone": "ruq"},
    "gallbladder":          {"extent": (0.025, 0.030, 0.025), "volume_ml": 50,
                             "surgical_zone": "ruq"},
    "bile_duct":            {"extent": (0.015, 0.040, 0.010), "volume_ml": 5,
                             "surgical_zone": "porta_hepatis"},
    "pancreas":             {"extent": (0.080, 0.020, 0.040), "volume_ml": 80,
                             "surgical_zone": "retroperitoneum"},

    # ════════════════════════════════════════════
    # RENAL / URINARY
    # ════════════════════════════════════════════
    "r_kidney":             {"extent": (0.040, 0.050, 0.030), "volume_ml": 150,
                             "surgical_zone": "right_retroperitoneum"},
    "l_kidney":             {"extent": (0.040, 0.050, 0.030), "volume_ml": 150,
                             "surgical_zone": "left_retroperitoneum"},
    "r_ureter":             {"extent": (0.005, 0.150, 0.020), "volume_ml": 5,
                             "surgical_zone": "right_retroperitoneum"},
    "l_ureter":             {"extent": (0.005, 0.150, 0.020), "volume_ml": 5,
                             "surgical_zone": "left_retroperitoneum"},
    "bladder":              {"extent": (0.050, 0.040, 0.040), "volume_ml": 500,
                             "surgical_zone": "pelvis"},
    "urethra":              {"extent": (0.010, 0.030, 0.015), "volume_ml": 5,
                             "surgical_zone": "perineum"},

    # ════════════════════════════════════════════
    # ENDOCRINE
    # ════════════════════════════════════════════
    "thyroid":              {"extent": (0.040, 0.020, 0.020), "volume_ml": 25,
                             "surgical_zone": "anterior_neck"},
    "pituitary":            {"extent": (0.005, 0.005, 0.005), "volume_ml": 1,
                             "surgical_zone": "sella_turcica"},
    "r_adrenal":            {"extent": (0.020, 0.020, 0.015), "volume_ml": 5,
                             "surgical_zone": "right_retroperitoneum"},
    "l_adrenal":            {"extent": (0.020, 0.020, 0.015), "volume_ml": 5,
                             "surgical_zone": "left_retroperitoneum"},

    # ════════════════════════════════════════════
    # HEMATOLOGIC / LYMPHATIC
    # ════════════════════════════════════════════
    "spleen":               {"extent": (0.050, 0.040, 0.040), "volume_ml": 150,
                             "surgical_zone": "luq"},
    "bone_marrow":          {"extent": (0.150, 0.300, 0.080), "volume_ml": 2500,
                             "surgical_zone": "skeletal"},
    "thoracic_duct":        {"extent": (0.005, 0.250, 0.010), "volume_ml": 8,
                             "surgical_zone": "posterior_mediastinum"},
    "right_lymphatic_duct": {"extent": (0.005, 0.020, 0.005), "volume_ml": 1,
                             "surgical_zone": "thoracic_inlet"},
    "cisterna_chyli":       {"extent": (0.010, 0.020, 0.010), "volume_ml": 3,
                             "surgical_zone": "retroperitoneum"},
    "cerv_ln":              {"extent": (0.040, 0.040, 0.030), "volume_ml": 20,
                             "surgical_zone": "neck"},
    "axil_ln":              {"extent": (0.030, 0.040, 0.030), "volume_ml": 20,
                             "surgical_zone": "axilla"},
    "med_ln":               {"extent": (0.030, 0.030, 0.040), "volume_ml": 20,
                             "surgical_zone": "mediastinum"},
    "mes_ln":               {"extent": (0.080, 0.080, 0.060), "volume_ml": 30,
                             "surgical_zone": "mesentery"},
    "ing_ln":               {"extent": (0.030, 0.020, 0.030), "volume_ml": 15,
                             "surgical_zone": "inguinal"},

    # ════════════════════════════════════════════
    # REPRODUCTIVE
    # ════════════════════════════════════════════
    "uterus":               {"extent": (0.040, 0.040, 0.030), "volume_ml": 80,
                             "surgical_zone": "pelvis_female"},
    "ovaries":              {"extent": (0.030, 0.020, 0.020), "volume_ml": 6,
                             "surgical_zone": "pelvis_female"},
    "prostate":             {"extent": (0.030, 0.030, 0.030), "volume_ml": 25,
                             "surgical_zone": "pelvis_male"},

    # ════════════════════════════════════════════
    # MSK (regions)
    # ════════════════════════════════════════════
    "lower_limbs":          {"extent": (0.150, 0.300, 0.080), "volume_ml": 25000,
                             "surgical_zone": "lower_limb"},
    "upper_limbs":          {"extent": (0.250, 0.150, 0.100), "volume_ml": 8000,
                             "surgical_zone": "upper_limb"},
    "l_arm":                {"extent": (0.080, 0.250, 0.080), "volume_ml": 4000,
                             "surgical_zone": "left_arm"},
    "r_arm":                {"extent": (0.080, 0.250, 0.080), "volume_ml": 4000,
                             "surgical_zone": "right_arm"},
    "jaw":                  {"extent": (0.060, 0.040, 0.040), "volume_ml": 100,
                             "surgical_zone": "face"},
    "r_shoulder":           {"extent": (0.040, 0.040, 0.040), "volume_ml": 200,
                             "surgical_zone": "shoulder"},
    "l_shoulder":           {"extent": (0.040, 0.040, 0.040), "volume_ml": 200,
                             "surgical_zone": "shoulder"},
    "epigastrium":          {"extent": (0.050, 0.050, 0.050), "volume_ml": 0,
                             "surgical_zone": "epigastric"},
    "skin":                 {"extent": (0.200, 0.300, 0.200), "volume_ml": 4000,
                             "surgical_zone": "integument"},
}


# Try to load from JSON (single source of truth), fall back to inline if missing
try:
    from .knowledge_loader import get_kb
    _kb = get_kb()
    ORGAN_GEOMETRY = _kb.organ_geometry or _FALLBACK_ORGAN_GEOMETRY
except (ImportError, Exception):
    try:
        from knowledge_loader import get_kb
        _kb = get_kb()
        ORGAN_GEOMETRY = _kb.organ_geometry or _FALLBACK_ORGAN_GEOMETRY
    except Exception:
        ORGAN_GEOMETRY = _FALLBACK_ORGAN_GEOMETRY


def get_geometry(organ_name: str) -> dict:
    """Return geometry dict for an organ, with safe defaults."""
    return ORGAN_GEOMETRY.get(organ_name, {
        "extent":        (0.015, 0.015, 0.015),
        "volume_ml":     5,
        "surgical_zone": "unknown",
    })


def get_organs_in_surgical_zone(zone: str) -> list:
    """List all organs in a surgical zone (e.g. 'rlq', 'porta_hepatis')."""
    return [name for name, geom in ORGAN_GEOMETRY.items()
            if geom.get("surgical_zone") == zone]


def total_volume_organs(organs: list) -> float:
    """Sum volumes of multiple organs in mL."""
    return sum(get_geometry(o).get("volume_ml", 0) for o in organs)