"""
NEXUS 
═══════════════════════════════════════════════════════════════
 + 

:
  1. :  →  →  →  → 
  2. :  →  → 
  3. : DVT →  →  → 
  4. :    → T1-T4 → 
  5. :  → /
  6. :  →  → 
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class Organ:
    name: str
    system: str
    region: str
    position: str
    pos_3d: Tuple[float, float, float] = (0, 0, 0)
    functions: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    # ── Hierarchy fields (Stage 1 expansion) ─────────────────────
    # parent: which organ this is a sub-structure of (None for top-level).
    # children: cached list of sub-structures (populated by _build_hierarchy).
    # level: depth in tree (0 = top-level system, 1 = major organ, etc.).
    # Backward compatible: existing 231 organs default to parent=None, level=0.
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    level: int = 0


@dataclass
class Connection:
    source: str
    target: str
    conn_type: str
    direction: str = "antegrade"
    weight: float = 1.0
    desc: str = ""


class AnatomyAtlas:

    def __init__(self):
        self.organs: Dict[str, Organ] = {}
        self.connections: List[Connection] = []
        self._fwd: Dict[str, List[Connection]] = defaultdict(list)
        self._rev: Dict[str, List[Connection]] = defaultdict(list)
        self._alias: Dict[str, str] = {}
        self._build()
        print(f"[ANATOMY] {len(self.organs)} organs, {len(self.connections)} connections")

    # ═══════════════════════════════════════════════
    #  Internal builders
    # ═══════════════════════════════════════════════

    def _o(self, name, system, region, pos, p3d, funcs=None, aliases=None):
        self.organs[name] = Organ(name, system, region, pos, p3d, funcs or [], aliases or [])

    def _oh(self, name, system, region, pos, p3d, funcs=None, aliases=None,
            parent=None, level=0):
        """Hierarchical organ: same as _o but with parent pointer and depth level.
        Used for tree-structured systems (e.g. nervous system) where organs have
        parent→child relationships beyond simple connections.
        """
        self.organs[name] = Organ(name, system, region, pos, p3d,
                                  funcs or [], aliases or [],
                                  parent=parent, level=level)

    def _c(self, src, tgt, ct, d="antegrade", desc="", w=1.0):
        c = Connection(src, tgt, ct, d, w, desc)
        self.connections.append(c)
        self._fwd[src].append(c)
        self._rev[tgt].append(c)
        if d == "bidirectional":
            c2 = Connection(tgt, src, ct, "bidirectional", w, desc)
            self.connections.append(c2)
            self._fwd[tgt].append(c2)
            self._rev[src].append(c2)

    def _build(self):
        self._build_organs()
        self._build_expanded_vessels()
        self._build_systemic_arteries()
        self._build_systemic_veins()
        self._build_portal()
        self._build_pulmonary()
        self._build_gi()
        self._build_urinary()
        self._build_biliary()
        self._build_neural()
        self._build_referred_pain()
        self._build_lymph()
        self._build_adjacent()
        self._build_blood_connections()   # blood ↔ all systems
        self._build_cns_hierarchy()       # Stage 1: central nervous system tree
        self._build_hierarchy_index()     # populate children[] from parent pointers
        # alias index
        for n, org in self.organs.items():
            self._alias[n.lower()] = n
            self._alias[n.replace("_"," ").lower()] = n
            for a in org.aliases:
                self._alias[a.lower()] = n

    # ───── ORGANS ─────

    def _build_organs(self):
        o = self._o
        # ── Cardiovascular ──
        o("heart","cardiovascular","thorax","midline",(0,.35,.1),["pump_blood"],[""])
        o("right_atrium","cardiovascular","thorax","right",(.05,.38,.05),["receive_venous"],[""])
        o("right_ventricle","cardiovascular","thorax","right",(.05,.32,.1),["pump_to_lungs"],[""])
        o("left_atrium","cardiovascular","thorax","left",(-.05,.38,-.05),["receive_oxy"],[""])
        o("left_ventricle","cardiovascular","thorax","left",(-.05,.32,.1),["pump_to_body"],[""])
        o("aorta","cardiovascular","thorax","midline",(0,.35,-.05),["main_artery"],[""])
        o("svc","cardiovascular","thorax","right",(.06,.42,0),[],[""])
        o("ivc","cardiovascular","abdomen","midline",(.04,.05,-.1),[],[""])
        o("coronary_aa","cardiovascular","thorax","midline",(0,.36,.15),[],[""])
        o("coronary_sinus","cardiovascular","thorax","midline",(0,.34,-.05),[],[""])
        o("pulm_aa","cardiovascular","thorax","midline",(.03,.4,.05),[],[""])
        o("pulm_vv","cardiovascular","thorax","midline",(-.03,.4,-.05),[],[""])
        o("portal_vein","cardiovascular","abdomen","midline",(.06,.1,-.02),["gi_blood_to_liver"],[""])
        o("hepatic_vv","cardiovascular","abdomen","right",(.1,.18,-.05),[],[""])
        o("splenic_v","cardiovascular","abdomen","left",(-.08,.1,-.03),[],[""])
        o("smv","cardiovascular","abdomen","midline",(.04,.02,-.02),[],[""])
        o("imv","cardiovascular","abdomen","left",(-.06,-.02,-.02),[],[""])
        o("celiac","cardiovascular","abdomen","midline",(0,.15,-.08),[],[""])
        o("sma","cardiovascular","abdomen","midline",(.02,.08,-.08),[],[""])
        o("ima","cardiovascular","abdomen","midline",(-.02,-.02,-.08),[],[""])
        o("hepatic_a","cardiovascular","abdomen","right",(.08,.14,0),[],[""])
        o("r_renal_a","cardiovascular","abdomen","right",(.08,.06,-.1),[],[""])
        o("l_renal_a","cardiovascular","abdomen","left",(-.08,.06,-.1),[],[""])
        o("r_renal_v","cardiovascular","abdomen","right",(.09,.05,-.08),[],[])

        # ── Major Arteries (systemic tree) ──
        o("aortic_arch","cardiovascular","thorax","midline",(0,.45,-.03),["arch"],[])
        o("thoracic_aorta","cardiovascular","thorax","midline",(0,.3,-.08),["descending"],[])
        o("abdominal_aorta","cardiovascular","abdomen","midline",(0,.1,-.1),["abdominal"],[])
        o("brachiocephalic_a","cardiovascular","thorax","right",(.06,.48,-.02),[],[])
        o("r_carotid_a","cardiovascular","neck","right",(.06,.6,-.02),["brain_supply"],[])
        o("l_carotid_a","cardiovascular","neck","left",(-.06,.6,-.02),["brain_supply"],[])
        o("r_subclavian_a","cardiovascular","thorax","right",(.15,.48,0),["arm_supply"],[])
        o("l_subclavian_a","cardiovascular","thorax","left",(-.15,.48,0),["arm_supply"],[])
        o("r_vertebral_a","cardiovascular","neck","right",(.04,.62,-.06),["brainstem_supply"],[])
        o("l_vertebral_a","cardiovascular","neck","left",(-.04,.62,-.06),["brainstem_supply"],[])
        o("basilar_a","cardiovascular","head","midline",(0,.72,-.08),["posterior_brain"],[])
        o("r_brachial_a","cardiovascular","limbs","right",(.25,.35,.05),[],[])
        o("l_brachial_a","cardiovascular","limbs","left",(-.25,.35,.05),[],[])
        o("r_radial_a","cardiovascular","limbs","right",(.28,.2,.08),[],[])
        o("l_radial_a","cardiovascular","limbs","left",(-.28,.2,.08),[],[])
        o("r_iliac_a","cardiovascular","pelvis","right",(.08,-.12,-.08),[],[])
        o("l_iliac_a","cardiovascular","pelvis","left",(-.08,-.12,-.08),[],[])
        o("r_femoral_a","cardiovascular","limbs","right",(.1,-.25,.05),["leg_supply"],[])
        o("l_femoral_a","cardiovascular","limbs","left",(-.1,-.25,.05),["leg_supply"],[])
        o("r_popliteal_a","cardiovascular","limbs","right",(.1,-.4,-.02),[],[])
        o("l_popliteal_a","cardiovascular","limbs","left",(-.1,-.4,-.02),[],[])
        o("r_tibial_a","cardiovascular","limbs","right",(.1,-.55,.02),[],[])
        o("l_tibial_a","cardiovascular","limbs","left",(-.1,-.55,.02),[],[])

        # ── Major Veins (systemic return) ──
        o("r_jugular_v","cardiovascular","neck","right",(.07,.58,.02),["head_drain"],[])
        o("l_jugular_v","cardiovascular","neck","left",(-.07,.58,.02),["head_drain"],[])
        o("r_subclavian_v","cardiovascular","thorax","right",(.14,.47,.02),[],[])
        o("l_subclavian_v","cardiovascular","thorax","left",(-.14,.47,.02),[],[])
        o("r_brachiocephalic_v","cardiovascular","thorax","right",(.08,.46,.01),[],[])
        o("l_brachiocephalic_v","cardiovascular","thorax","left",(-.08,.46,.01),[],[])
        o("azygos_v","cardiovascular","thorax","right",(.05,.3,-.12),["thorax_drain"],[])
        o("r_femoral_v","cardiovascular","limbs","right",(.11,-.25,.02),[],[])
        o("l_femoral_v","cardiovascular","limbs","left",(-.11,-.25,.02),[],[])
        o("r_iliac_v","cardiovascular","pelvis","right",(.09,-.12,-.06),[],[])
        o("l_iliac_v","cardiovascular","pelvis","left",(-.09,-.12,-.06),[],[])
        o("r_saphenous_v","cardiovascular","limbs","right",(.12,-.4,.08),["dvt_risk"],[])
        o("l_saphenous_v","cardiovascular","limbs","left",(-.12,-.4,.08),["dvt_risk"],[])
        o("r_popliteal_v","cardiovascular","limbs","right",(.11,-.4,-.01),[],[])
        o("l_popliteal_v","cardiovascular","limbs","left",(-.11,-.4,-.01),[],[])
        o("l_renal_v","cardiovascular","abdomen","left",(-.09,.05,-.08),[],[""])


        # ── Cerebral Circulation (Circle of Willis + dural sinuses) ──
        o("internal_carotid_a","cardiovascular","head","right",(.05,.7,-.04),["brain_supply_anterior"],[])
        o("l_internal_carotid_a","cardiovascular","head","left",(-.05,.7,-.04),["brain_supply_anterior"],[])
        o("external_carotid_a","cardiovascular","head","right",(.07,.65,.02),["face_scalp_supply"],[])
        o("l_external_carotid_a","cardiovascular","head","left",(-.07,.65,.02),["face_scalp_supply"],[])
        o("anterior_cerebral_a","cardiovascular","head","midline",(0,.78,-.05),["frontal_lobe","aca"],[])
        o("middle_cerebral_a","cardiovascular","head","right",(.04,.76,-.05),["mca","stroke_common"],[])
        o("l_middle_cerebral_a","cardiovascular","head","left",(-.04,.76,-.05),["mca","stroke_common"],[])
        o("posterior_cerebral_a","cardiovascular","head","midline",(0,.74,-.1),["pca","occipital"],[])
        o("anterior_communicating_a","cardiovascular","head","midline",(0,.78,-.06),["acomm","aneurysm_site"],[])
        o("posterior_communicating_a","cardiovascular","head","right",(.03,.75,-.08),["pcomm"],[])
        o("ophthalmic_a","cardiovascular","head","right",(.04,.73,.02),["eye_supply"],[])
        o("superior_sagittal_sinus","cardiovascular","head","midline",(0,.82,-.02),["dural_sinus","cvt_site"],[])
        o("transverse_sinus","cardiovascular","head","right",(.06,.78,-.1),["dural_sinus"],[])
        o("sigmoid_sinus","cardiovascular","head","right",(.07,.7,-.1),["dural_sinus","drains_to_jugular"],[])
        o("cavernous_sinus","cardiovascular","head","midline",(0,.72,-.05),["dural_sinus","cn_3_4_5_6"],[])

        # ── Coronary Arteries (MI territories) ──
        o("left_main_coronary","cardiovascular","thorax","midline",(0,.36,.12),["lmca","critical"],[])
        o("lad_a","cardiovascular","thorax","midline",(-.02,.34,.13),["widow_maker","anterior_wall"],[])
        o("lcx_a","cardiovascular","thorax","left",(-.04,.35,.1),["lateral_wall"],[])
        o("rca_a","cardiovascular","thorax","right",(.04,.35,.1),["inferior_wall","sa_node_av_node"],[])

        # ── Pulmonary Detailed ──
        o("r_pulm_a","cardiovascular","thorax","right",(.05,.4,.05),["right_lung_supply"],[])
        o("l_pulm_a","cardiovascular","thorax","left",(-.05,.4,.05),["left_lung_supply"],[])
        o("r_upper_pulm_v","cardiovascular","thorax","right",(.06,.42,-.03),[],[])
        o("r_lower_pulm_v","cardiovascular","thorax","right",(.06,.36,-.03),[],[])
        o("l_upper_pulm_v","cardiovascular","thorax","left",(-.06,.42,-.03),[],[])
        o("l_lower_pulm_v","cardiovascular","thorax","left",(-.06,.36,-.03),[],[])
        o("bronchial_a","cardiovascular","thorax","midline",(0,.38,-.02),["bronchi_supply"],[])

        # ── Abdominal Visceral Detail ──
        o("splenic_a","cardiovascular","abdomen","left",(-.07,.1,-.04),["spleen_supply"],[])
        o("common_hepatic_a","cardiovascular","abdomen","right",(.06,.13,-.02),["liver_stomach_supply"],[])
        o("proper_hepatic_a","cardiovascular","abdomen","right",(.08,.14,.02),["liver_supply"],[])
        o("left_gastric_a","cardiovascular","abdomen","left",(-.04,.14,-.02),["stomach_supply"],[])
        o("right_gastric_a","cardiovascular","abdomen","right",(.04,.14,-.02),["stomach_supply"],[])
        o("gastroduodenal_a","cardiovascular","abdomen","right",(.06,.1,.02),["pud_bleed_source"],[])
        o("cystic_a","cardiovascular","abdomen","right",(.1,.1,.1),["gallbladder_supply"],[])
        o("ileocolic_a","cardiovascular","abdomen","right",(.13,-.05,.02),["appendix_supply"],[])
        o("right_colic_a","cardiovascular","abdomen","right",(.14,.02,.02),[],[])
        o("middle_colic_a","cardiovascular","abdomen","midline",(0,.06,.06),[],[])
        o("left_colic_a","cardiovascular","abdomen","left",(-.14,.02,.02),[],[])
        o("sigmoid_aa","cardiovascular","pelvis","left",(-.1,-.12,.02),[],[])
        o("superior_rectal_a","cardiovascular","pelvis","midline",(0,-.18,-.04),["rectum_supply"],[])

        # ── Portal System Detail ──
        o("left_gastric_v","cardiovascular","abdomen","left",(-.04,.13,-.03),["varices_origin"],[])
        o("paraumbilical_vv","cardiovascular","abdomen","midline",(0,0,.1),["caput_medusae_cirrhosis"],[])
        o("cystic_v","cardiovascular","abdomen","right",(.1,.1,.08),[],[])

        # ── Pelvic ──
        o("internal_iliac_a","cardiovascular","pelvis","right",(.06,-.13,-.06),["pelvic_organs"],[])
        o("l_internal_iliac_a","cardiovascular","pelvis","left",(-.06,-.13,-.06),["pelvic_organs"],[])
        o("external_iliac_a","cardiovascular","pelvis","right",(.09,-.13,-.04),["leads_to_femoral"],[])
        o("l_external_iliac_a","cardiovascular","pelvis","left",(-.09,-.13,-.04),["leads_to_femoral"],[])
        o("uterine_a","cardiovascular","pelvis","midline",(0,-.16,-.05),["pregnancy"],[])
        o("ovarian_a","cardiovascular","pelvis","right",(.05,-.1,-.08),[],[])
        o("testicular_a","cardiovascular","pelvis","right",(.05,-.15,.05),[],[])
        o("vesical_aa","cardiovascular","pelvis","midline",(0,-.18,-.02),["bladder_supply"],[])
        o("obturator_a","cardiovascular","pelvis","right",(.07,-.18,-.02),[],[])

        # ── Upper Limb Detail ──
        o("axillary_a","cardiovascular","limbs","right",(.18,.42,.04),[],[])
        o("l_axillary_a","cardiovascular","limbs","left",(-.18,.42,.04),[],[])
        o("deep_brachial_a","cardiovascular","limbs","right",(.24,.38,.04),[],[])
        o("ulnar_a","cardiovascular","limbs","right",(.27,.21,.07),[],[])
        o("l_ulnar_a","cardiovascular","limbs","left",(-.27,.21,.07),[],[])
        o("palmar_arch_aa","cardiovascular","limbs","right",(.3,.15,.08),[],[])
        o("l_palmar_arch_aa","cardiovascular","limbs","left",(-.3,.15,.08),[],[])
        o("axillary_v","cardiovascular","limbs","right",(.17,.41,.05),[],[])
        o("l_axillary_v","cardiovascular","limbs","left",(-.17,.41,.05),[],[])
        o("cephalic_v","cardiovascular","limbs","right",(.26,.3,.08),["lateral_arm_v"],[])
        o("basilic_v","cardiovascular","limbs","right",(.22,.3,.06),["medial_arm_v"],[])
        o("median_cubital_v","cardiovascular","limbs","right",(.24,.28,.07),["phlebotomy_site"],[])

        # ── Lower Limb Detail ──
        o("deep_femoral_a","cardiovascular","limbs","right",(.09,-.27,.03),["profunda"],[])
        o("l_deep_femoral_a","cardiovascular","limbs","left",(-.09,-.27,.03),["profunda"],[])
        o("anterior_tibial_a","cardiovascular","limbs","right",(.1,-.5,.04),[],[])
        o("l_anterior_tibial_a","cardiovascular","limbs","left",(-.1,-.5,.04),[],[])
        o("posterior_tibial_a","cardiovascular","limbs","right",(.1,-.5,-.02),[],[])
        o("l_posterior_tibial_a","cardiovascular","limbs","left",(-.1,-.5,-.02),[],[])
        o("peroneal_a","cardiovascular","limbs","right",(.12,-.5,-.02),[],[])
        o("dorsalis_pedis_a","cardiovascular","limbs","right",(.1,-.62,.04),["foot_pulse"],[])
        o("l_dorsalis_pedis_a","cardiovascular","limbs","left",(-.1,-.62,.04),["foot_pulse"],[])
        o("deep_femoral_v","cardiovascular","limbs","right",(.1,-.27,.02),[],[])
        o("great_saphenous_v","cardiovascular","limbs","right",(.13,-.4,.09),["longest_v","dvt_site"],[])
        o("l_great_saphenous_v","cardiovascular","limbs","left",(-.13,-.4,.09),["longest_v","dvt_site"],[])
        o("small_saphenous_v","cardiovascular","limbs","right",(.11,-.5,-.04),[],[])
        o("soleal_vv","cardiovascular","limbs","right",(.1,-.5,-.03),["dvt_origin"],[])
        o("gastrocnemius_vv","cardiovascular","limbs","right",(.11,-.45,-.02),["dvt_origin"],[])
        o("perforator_vv","cardiovascular","limbs","right",(.12,-.4,.02),[],[])

        # ── Major Lymphatic Trunks ──
        o("thoracic_duct","lymphatic","thorax","left",(-.02,.3,-.13),["main_lymph_drain"],[])
        o("right_lymphatic_duct","lymphatic","thorax","right",(.02,.45,.02),["right_upper_drain"],[])
        o("cisterna_chyli","lymphatic","abdomen","midline",(0,.05,-.13),["chyle_collection"],[])

        # ── Respiratory ──
        o("trachea","respiratory","neck","midline",(0,.55,.05),["airway"],[""])
        o("r_bronchus","respiratory","thorax","right",(.08,.42,0),[],[""])
        o("l_bronchus","respiratory","thorax","left",(-.08,.42,0),[],[""])
        o("r_lung","respiratory","thorax","right",(.15,.35,0),["gas_exchange"],[""])
        o("l_lung","respiratory","thorax","left",(-.15,.35,0),["gas_exchange"],[""])
        o("lungs","respiratory","thorax","bilateral",(0,.35,0),["gas_exchange","filter"],["",""])
        o("diaphragm","respiratory","thorax","midline",(0,.2,0),["breathing"],["",""])
        o("pleura","respiratory","thorax","bilateral",(.18,.35,.05),[],[""])

        # ── GI Tract ──
        o("mouth","gi","head","midline",(0,.75,.2),[],[""])
        o("pharynx","gi","neck","midline",(0,.65,.05),[],[""])
        o("esophagus","gi","thorax","midline",(0,.45,-.1),["food_transport"],["",""])
        o("stomach","gi","abdomen","left",(-.1,.12,.1),["digestion","acid"],[""])
        o("duodenum","gi","abdomen","right",(.06,.06,.05),["absorption"],[""])
        o("jejunum","gi","abdomen","left",(-.05,0,.05),["nutrient_absorption"],[""])
        o("ileum","gi","abdomen","right",(.05,-.05,.05),["absorption"],[""])
        o("cecum","gi","abdomen","right",(.15,-.1,.05),[],[""])
        o("appendix","gi","abdomen","right",(.17,-.13,.08),["immune"],[""])
        o("asc_colon","gi","abdomen","right",(.16,0,0),[],[""])
        o("trans_colon","gi","abdomen","midline",(0,.08,.08),[],[""])
        o("desc_colon","gi","abdomen","left",(-.16,0,0),[],[""])
        o("sig_colon","gi","pelvis","left",(-.12,-.15,.05),[],[""])
        o("rectum","gi","pelvis","midline",(0,-.22,-.05),[],[""])
        o("anus","gi","pelvis","midline",(0,-.28,.05),[],[""])

        # ── Hepatobiliary ──
        o("liver","hepatobiliary","abdomen","right",(.12,.15,.08),["metabolism","detox","bile","filter_portal"],["",""])
        o("gallbladder","hepatobiliary","abdomen","right",(.1,.1,.12),["store_bile"],[""])
        o("bile_duct","hepatobiliary","abdomen","right",(.08,.08,.08),[],["",""])
        o("pancreas","hepatobiliary","abdomen","midline",(0,.08,-.05),["enzymes","insulin"],["",""])
        o("spleen","hematologic","abdomen","left",(-.16,.12,-.05),["filter","immune"],["",""])

        # ── Renal / Urinary ──
        o("r_kidney","renal","abdomen","right",(.12,.05,-.12),["filtration"],[""])
        o("l_kidney","renal","abdomen","left",(-.12,.05,-.12),["filtration"],[""])
        o("r_ureter","renal","abdomen","right",(.1,-.08,-.1),[],[""])
        o("l_ureter","renal","abdomen","left",(-.1,-.08,-.1),[],[""])
        o("bladder","renal","pelvis","midline",(0,-.2,.05),["urine_storage"],[""])
        o("urethra","renal","pelvis","midline",(0,-.28,.1),[],[""])
        o("r_adrenal","endocrine","abdomen","right",(.1,.1,-.12),[],[""])
        o("l_adrenal","endocrine","abdomen","left",(-.1,.1,-.12),[],[""])

        # ── Neurologic ──
        o("brain","neurologic","head","midline",(0,.85,0),["cognition","motor","sensory"],["",""])
        o("cerebellum","neurologic","head","midline",(0,.78,-.1),["coordination"],[""])
        o("brainstem","neurologic","head","midline",(0,.72,-.05),["vital_functions"],[""])
        o("spinal_cord","neurologic","spine","midline",(0,.3,-.15),["relay"],[""])
        o("meninges","neurologic","head","midline",(0,.86,.02),[],[""])
        o("vagus_n","neurologic","neck","bilateral",(.05,.55,-.05),["parasympathetic"],[""])
        o("phrenic_n","neurologic","neck","bilateral",(.06,.5,-.03),[],[""])

        # ── Endocrine ──
        o("thyroid","endocrine","neck","midline",(0,.58,.1),[],[""])
        o("pituitary","endocrine","head","midline",(0,.75,0),[],[""])

        # ── Limbs / Regions ──
        o("lower_limbs","msk","limbs","bilateral",(.08,-.5,.05),[],[""])
        o("upper_limbs","msk","limbs","bilateral",(.22,.3,.1),[],[""])
        o("l_arm","msk","limbs","left",(-.25,.25,.1),[],[""])
        o("r_arm","msk","limbs","right",(.25,.25,.1),[],[""])
        o("jaw","msk","head","midline",(0,.7,.15),[],[""])
        o("r_shoulder","msk","thorax","right",(.22,.45,.05),[],[""])
        o("l_shoulder","msk","thorax","left",(-.22,.45,.05),[],[""])
        o("epigastrium","region","abdomen","midline",(0,.15,.15),[],[""])
        o("bone_marrow","hematologic","whole","bilateral",(0,.1,0),["blood_production"],[""])
        o("skin","integumentary","whole","bilateral",(.2,.3,.2),[],[""])

        # ── Reproductive ──
        o("uterus","reproductive","pelvis","midline",(0,-.18,0),[],[""])
        o("ovaries","reproductive","pelvis","bilateral",(.08,-.15,0),[],[""])
        o("prostate","reproductive","pelvis","midline",(0,-.23,.02),[],[""])

        # ── Lymph nodes ──
        o("cerv_ln","lymphatic","neck","bilateral",(.08,.6,.08),[],[""])
        o("axil_ln","lymphatic","thorax","bilateral",(.2,.4,.05),[],[""])
        o("med_ln","lymphatic","thorax","midline",(0,.35,-.08),[],[""])
        o("mes_ln","lymphatic","abdomen","midline",(0,.05,-.05),[],[""])
        o("ing_ln","lymphatic","pelvis","bilateral",(.12,-.25,.1),[],[""])

        # ════════════════════════════════════════════════════════════════
        # ── Spatial-Fingerprint Expansion (24 organs) ──
        # Added to provide spatial proof for previously-missing registry
        # organs. system labels chosen for evidence_gate compatibility.
        # ════════════════════════════════════════════════════════════════

        # ── Sensory (NEW system="sensory") ──
        o("eye",         "sensory","head","bilateral",(.04,.78,.15), ["vision_capture"],[])
        o("retina",      "sensory","head","bilateral",(.04,.78,.12), ["photoreception"],[])
        o("middle_ear",  "sensory","head","bilateral",(.08,.74,0),   ["hearing_conduct"],[])

        # ── Peripheral nerves (neurologic) ──
        o("median_nerve",         "neurologic","limbs","bilateral",(.27,.18,.08), ["hand_motor_sensory"],[])
        o("ulnar_nerve",          "neurologic","limbs","bilateral",(.27,.18,.04), ["hand_motor_sensory"],[])
        o("tibial_nerve",         "neurologic","limbs","bilateral",(.08,-.55,0),  ["foot_motor_sensory"],[])
        o("common_peroneal_nerve","neurologic","limbs","bilateral",(.09,-.42,-.02),["foot_dorsiflexion"],[])
        o("brachial_plexus",      "neurologic","neck", "bilateral",(.12,.50,.02), ["upper_limb_nerves"],[])
        o("facial_nerve",         "neurologic","head", "bilateral",(.06,.70,.08), ["facial_motor"],[])
        o("trigeminal_nerve",     "neurologic","head", "bilateral",(.04,.74,.05), ["face_sensory"],[])

        # ── Joints (6, msk) for multi-joint diseases (Gout, RA) ──
        o("mtp_joint",  "msk","limbs","bilateral",(.05,-.65,.02), ["1st_metatarsophalangeal"],[])
        o("mcp_joint",  "msk","limbs","bilateral",(.30,.15,.06),  ["metacarpophalangeal"],[])
        o("pip_joint",  "msk","limbs","bilateral",(.32,.14,.06),  ["proximal_interphalangeal"],[])
        o("wrist_joint","msk","limbs","bilateral",(.27,.20,.05),  ["wrist_articulation"],[])
        o("knee_joint", "msk","limbs","bilateral",(.10,-.35,.02), ["knee_articulation"],[])
        o("ankle_joint","msk","limbs","bilateral",(.09,-.55,.02), ["ankle_articulation"],[])

        # ── Other musculoskeletal ──
        o("piriformis_muscle",       "msk","pelvis","bilateral",(.07,-.12,-.08),["hip_external_rotation"],[])
        o("supraspinatus_muscle",    "msk","limbs", "bilateral",(.18,.42,-.05), ["shoulder_abduction"],[])
        o("iliotibial_band",         "msk","limbs", "bilateral",(.13,-.30,0),   ["lateral_knee_stability"],[])
        o("fascial_compartment_leg", "msk","limbs", "bilateral",(.08,-.45,0),   ["compartment_pressure"],[])

        # ── Visceral ──
        o("breast",          "reproductive","thorax", "bilateral",(.12,.30,.18), ["lactation"],[])
        o("pancreas_islets", "endocrine",   "abdomen","midline",  (-.04,.10,-.05),["insulin_secretion"],[])

        # ── Systemic (blood as placeholder organ) ──
        # blood lives in hematologic system but physically connects to many organs
        # (see _build_blood_connections below).
        o("blood","hematologic","whole","bilateral",(0,.20,0),["o2_transport","hemostasis"],[])

    # ───── EXPANDED VESSEL CONNECTIONS ─────
    def _build_expanded_vessels(self):
        """Connections for the 75 new vessels added for clinical fidelity."""
        c = self._c

        # ── Cerebral arterial tree (Circle of Willis) ──
        c("r_carotid_a","internal_carotid_a","arterial",desc="bifurcation in neck")
        c("r_carotid_a","external_carotid_a","arterial",desc="face/scalp")
        c("l_carotid_a","l_internal_carotid_a","arterial")
        c("l_carotid_a","l_external_carotid_a","arterial")
        c("internal_carotid_a","middle_cerebral_a","arterial",desc="MCA — most common stroke")
        c("l_internal_carotid_a","l_middle_cerebral_a","arterial")
        c("internal_carotid_a","anterior_cerebral_a","arterial",desc="ACA")
        c("internal_carotid_a","ophthalmic_a","arterial",desc="eye supply")
        c("internal_carotid_a","posterior_communicating_a","arterial")
        c("anterior_cerebral_a","anterior_communicating_a","arterial")
        c("basilar_a","posterior_cerebral_a","arterial",desc="PCA — occipital")
        c("posterior_cerebral_a","posterior_communicating_a","arterial")
        c("middle_cerebral_a","brain","arterial",desc="MCA territory")
        c("l_middle_cerebral_a","brain","arterial",desc="MCA territory")
        c("anterior_cerebral_a","brain","arterial",desc="ACA territory")
        c("posterior_cerebral_a","brain","arterial",desc="PCA territory")

        # ── Cerebral venous drainage (dural sinuses) ──
        c("brain","superior_sagittal_sinus","venous",desc="superficial cortical drain")
        c("superior_sagittal_sinus","transverse_sinus","venous")
        c("transverse_sinus","sigmoid_sinus","venous")
        c("sigmoid_sinus","r_jugular_v","venous",desc="dural sinus to IJV")
        c("brain","cavernous_sinus","venous",desc="deep brain + face")
        c("cavernous_sinus","sigmoid_sinus","venous")

        # ── Coronary detail ──
        c("aorta","left_main_coronary","arterial",desc="LMCA from left coronary cusp")
        c("left_main_coronary","lad_a","arterial",desc="widow maker")
        c("left_main_coronary","lcx_a","arterial",desc="left circumflex")
        c("aorta","rca_a","arterial",desc="RCA from right coronary cusp")
        c("lad_a","heart","arterial",desc="anterior wall")
        c("lcx_a","heart","arterial",desc="lateral wall")
        c("rca_a","heart","arterial",desc="inferior wall + AV node")

        # ── Pulmonary detailed ──
        c("pulm_aa","r_pulm_a","arterial"); c("pulm_aa","l_pulm_a","arterial")
        c("r_pulm_a","r_lung","arterial"); c("l_pulm_a","l_lung","arterial")
        c("r_lung","r_upper_pulm_v","venous"); c("r_lung","r_lower_pulm_v","venous")
        c("l_lung","l_upper_pulm_v","venous"); c("l_lung","l_lower_pulm_v","venous")
        c("r_upper_pulm_v","left_atrium","venous"); c("r_lower_pulm_v","left_atrium","venous")
        c("l_upper_pulm_v","left_atrium","venous"); c("l_lower_pulm_v","left_atrium","venous")
        c("thoracic_aorta","bronchial_a","arterial",desc="bronchi nutrient supply")
        c("bronchial_a","r_bronchus","arterial"); c("bronchial_a","l_bronchus","arterial")

        # ── Abdominal visceral detail ──
        c("celiac","splenic_a","arterial"); c("splenic_a","spleen","arterial")
        c("celiac","common_hepatic_a","arterial")
        c("common_hepatic_a","proper_hepatic_a","arterial")
        c("proper_hepatic_a","liver","arterial",desc="hepatic arterial supply")
        c("common_hepatic_a","gastroduodenal_a","arterial",desc="PUD bleed source")
        c("gastroduodenal_a","duodenum","arterial"); c("gastroduodenal_a","stomach","arterial")
        c("celiac","left_gastric_a","arterial"); c("left_gastric_a","stomach","arterial")
        c("common_hepatic_a","right_gastric_a","arterial")
        c("proper_hepatic_a","cystic_a","arterial"); c("cystic_a","gallbladder","arterial")
        c("sma","ileocolic_a","arterial"); c("ileocolic_a","appendix","arterial",desc="appendiceal supply")
        c("ileocolic_a","cecum","arterial"); c("ileocolic_a","ileum","arterial")
        c("sma","right_colic_a","arterial"); c("right_colic_a","asc_colon","arterial")
        c("sma","middle_colic_a","arterial"); c("middle_colic_a","trans_colon","arterial")
        c("ima","left_colic_a","arterial"); c("left_colic_a","desc_colon","arterial")
        c("ima","sigmoid_aa","arterial"); c("sigmoid_aa","sig_colon","arterial")
        c("ima","superior_rectal_a","arterial"); c("superior_rectal_a","rectum","arterial")

        # ── Portal additions (varices/cirrhosis) ──
        c("stomach","left_gastric_v","venous"); c("left_gastric_v","portal_vein","venous")
        c("gallbladder","cystic_v","venous"); c("cystic_v","portal_vein","venous")
        c("liver","paraumbilical_vv","venous",desc="caput medusae in portal HTN")

        # ── Pelvic ──
        c("abdominal_aorta","internal_iliac_a","arterial")
        c("abdominal_aorta","l_internal_iliac_a","arterial")
        c("r_iliac_a","external_iliac_a","arterial")
        c("l_iliac_a","l_external_iliac_a","arterial")
        c("external_iliac_a","r_femoral_a","arterial",desc="becomes femoral at inguinal lig")
        c("l_external_iliac_a","l_femoral_a","arterial")
        c("internal_iliac_a","uterine_a","arterial")
        c("internal_iliac_a","vesical_aa","arterial"); c("vesical_aa","bladder","arterial")
        c("internal_iliac_a","obturator_a","arterial")
        c("abdominal_aorta","ovarian_a","arterial")
        c("abdominal_aorta","testicular_a","arterial")

        # ── Upper limb detail ──
        c("r_subclavian_a","axillary_a","arterial"); c("l_subclavian_a","l_axillary_a","arterial")
        c("axillary_a","r_brachial_a","arterial"); c("l_axillary_a","l_brachial_a","arterial")
        c("r_brachial_a","deep_brachial_a","arterial")
        c("r_brachial_a","ulnar_a","arterial"); c("l_brachial_a","l_ulnar_a","arterial")
        c("r_radial_a","palmar_arch_aa","arterial"); c("ulnar_a","palmar_arch_aa","arterial")
        c("l_radial_a","l_palmar_arch_aa","arterial"); c("l_ulnar_a","l_palmar_arch_aa","arterial")
        c("r_arm","cephalic_v","venous"); c("r_arm","basilic_v","venous")
        c("cephalic_v","median_cubital_v","venous"); c("basilic_v","median_cubital_v","venous")
        c("median_cubital_v","axillary_v","venous"); c("axillary_v","r_subclavian_v","venous")
        c("l_axillary_v","l_subclavian_v","venous")

        # ── Lower limb detail ──
        c("r_femoral_a","deep_femoral_a","arterial",desc="profunda femoris")
        c("l_femoral_a","l_deep_femoral_a","arterial")
        c("r_popliteal_a","anterior_tibial_a","arterial")
        c("r_popliteal_a","posterior_tibial_a","arterial")
        c("l_popliteal_a","l_anterior_tibial_a","arterial")
        c("l_popliteal_a","l_posterior_tibial_a","arterial")
        c("posterior_tibial_a","peroneal_a","arterial")
        c("anterior_tibial_a","dorsalis_pedis_a","arterial",desc="palpable foot pulse")
        c("l_anterior_tibial_a","l_dorsalis_pedis_a","arterial")
        c("r_femoral_v","deep_femoral_v","venous")
        c("r_saphenous_v","great_saphenous_v","venous")
        c("l_saphenous_v","l_great_saphenous_v","venous")
        c("great_saphenous_v","r_femoral_v","venous",desc="saphenofemoral junction")
        c("l_great_saphenous_v","l_femoral_v","venous")
        c("soleal_vv","r_popliteal_v","venous",desc="DVT origin in calf")
        c("gastrocnemius_vv","r_popliteal_v","venous",desc="DVT origin in calf")
        c("small_saphenous_v","r_popliteal_v","venous")
        c("perforator_vv","great_saphenous_v","venous",desc="superficial-deep")

        # ── Major lymphatic trunks ──
        c("cisterna_chyli","thoracic_duct","lymphatic",desc="origin of thoracic duct")
        c("thoracic_duct","l_brachiocephalic_v","lymphatic",desc="empties at left venous angle")
        c("right_lymphatic_duct","r_brachiocephalic_v","lymphatic",desc="empties at right venous angle")

    # ───── SYSTEMIC ARTERIES ─────
    def _build_systemic_arteries(self):
        c = self._c
        # Heart outflow
        c("left_ventricle","aorta","arterial",desc="LV ejection")
        c("aorta","aortic_arch","arterial"); c("aortic_arch","thoracic_aorta","arterial")
        c("thoracic_aorta","abdominal_aorta","arterial")
        # Coronary
        c("aorta","coronary_aa","arterial"); c("coronary_aa","heart","arterial")
        # Arch branches
        c("aortic_arch","brachiocephalic_a","arterial")
        c("brachiocephalic_a","r_carotid_a","arterial"); c("brachiocephalic_a","r_subclavian_a","arterial")
        c("aortic_arch","l_carotid_a","arterial"); c("aortic_arch","l_subclavian_a","arterial")
        # Carotids to brain
        c("r_carotid_a","brain","arterial",desc="anterior circulation")
        c("l_carotid_a","brain","arterial",desc="anterior circulation")
        # Vertebral-basilar
        c("r_subclavian_a","r_vertebral_a","arterial"); c("l_subclavian_a","l_vertebral_a","arterial")
        c("r_vertebral_a","basilar_a","arterial"); c("l_vertebral_a","basilar_a","arterial")
        c("basilar_a","brain","arterial",desc="posterior circulation")
        c("basilar_a","cerebellum","arterial"); c("basilar_a","brainstem","arterial")
        # Upper limbs
        c("r_subclavian_a","r_brachial_a","arterial"); c("l_subclavian_a","l_brachial_a","arterial")
        c("r_brachial_a","r_radial_a","arterial"); c("l_brachial_a","l_radial_a","arterial")
        c("r_brachial_a","r_arm","arterial"); c("l_brachial_a","l_arm","arterial")
        c("r_radial_a","r_arm","arterial"); c("l_radial_a","l_arm","arterial")
        # Abdominal branches
        c("abdominal_aorta","celiac","arterial")
        c("celiac","hepatic_a","arterial"); c("hepatic_a","liver","arterial",desc="hepatic arterial (25%)")
        c("celiac","stomach","arterial"); c("celiac","spleen","arterial")
        c("abdominal_aorta","sma","arterial")
        c("sma","duodenum","arterial"); c("sma","jejunum","arterial"); c("sma","ileum","arterial")
        c("sma","cecum","arterial"); c("sma","asc_colon","arterial"); c("sma","pancreas","arterial")
        c("abdominal_aorta","ima","arterial")
        c("ima","desc_colon","arterial"); c("ima","sig_colon","arterial"); c("ima","rectum","arterial")
        c("abdominal_aorta","r_renal_a","arterial"); c("r_renal_a","r_kidney","arterial")
        c("abdominal_aorta","l_renal_a","arterial"); c("l_renal_a","l_kidney","arterial")
        c("abdominal_aorta","r_adrenal","arterial"); c("abdominal_aorta","l_adrenal","arterial")
        # Lower limbs
        c("abdominal_aorta","r_iliac_a","arterial"); c("abdominal_aorta","l_iliac_a","arterial")
        c("r_iliac_a","r_femoral_a","arterial"); c("l_iliac_a","l_femoral_a","arterial")
        c("r_femoral_a","r_popliteal_a","arterial"); c("l_femoral_a","l_popliteal_a","arterial")
        c("r_popliteal_a","r_tibial_a","arterial"); c("l_popliteal_a","l_tibial_a","arterial")
        c("r_femoral_a","lower_limbs","arterial"); c("l_femoral_a","lower_limbs","arterial")

    # ───── SYSTEMIC VEINS ─────
    def _build_systemic_veins(self):
        c = self._c
        # Head/neck drainage
        c("brain","r_jugular_v","venous",desc="cerebral venous drain")
        c("brain","l_jugular_v","venous")
        c("r_jugular_v","r_brachiocephalic_v","venous"); c("l_jugular_v","l_brachiocephalic_v","venous")
        c("r_subclavian_v","r_brachiocephalic_v","venous"); c("l_subclavian_v","l_brachiocephalic_v","venous")
        c("r_brachiocephalic_v","svc","venous"); c("l_brachiocephalic_v","svc","venous")
        c("svc","right_atrium","venous",desc="SVC to RA")
        # Upper limbs
        c("r_arm","r_subclavian_v","venous"); c("l_arm","l_subclavian_v","venous")
        c("upper_limbs","svc","venous")
        # Coronary
        c("heart","coronary_sinus","venous"); c("coronary_sinus","right_atrium","venous")
        # Thorax
        c("azygos_v","svc","venous",desc="thorax drain to SVC")
        # Hepatic
        c("liver","hepatic_vv","venous"); c("hepatic_vv","ivc","venous")
        # Renal
        c("r_kidney","r_renal_v","venous"); c("r_renal_v","ivc","venous")
        c("l_kidney","l_renal_v","venous"); c("l_renal_v","ivc","venous")
        # Lower limbs (DVT pathway)
        c("lower_limbs","r_saphenous_v","venous"); c("lower_limbs","l_saphenous_v","venous")
        c("r_saphenous_v","r_femoral_v","venous"); c("l_saphenous_v","l_femoral_v","venous")
        c("r_popliteal_v","r_femoral_v","venous"); c("l_popliteal_v","l_femoral_v","venous")
        c("r_femoral_v","r_iliac_v","venous"); c("l_femoral_v","l_iliac_v","venous")
        c("r_iliac_v","ivc","venous"); c("l_iliac_v","ivc","venous")
        c("ivc","right_atrium","venous",desc="IVC to RA")
        # DVT -> PE pathway (critical clinical path)
        c("r_femoral_v","ivc","venous",desc="DVT embolism path")
        c("l_femoral_v","ivc","venous",desc="DVT embolism path")

    # ───── PORTAL SYSTEM ─────
    def _build_portal(self):
        c = self._c
        c("stomach","portal_vein","portal"); c("duodenum","portal_vein","portal")
        c("jejunum","smv","portal"); c("ileum","smv","portal")
        c("cecum","smv","portal"); c("asc_colon","smv","portal")
        c("smv","portal_vein","portal",desc="→")
        c("desc_colon","imv","portal"); c("sig_colon","imv","portal"); c("rectum","imv","portal")
        c("imv","splenic_v","portal"); c("spleen","splenic_v","portal"); c("pancreas","splenic_v","portal")
        c("splenic_v","portal_vein","portal",desc="→")
        c("portal_vein","liver","portal",desc="→(75%)")

    # ───── PULMONARY ─────
    def _build_pulmonary(self):
        c = self._c
        c("right_atrium","right_ventricle","cardiac",desc="")
        c("right_ventricle","pulm_aa","arterial",desc="")
        c("pulm_aa","r_lung","arterial"); c("pulm_aa","l_lung","arterial"); c("pulm_aa","lungs","arterial")
        c("r_lung","pulm_vv","venous"); c("l_lung","pulm_vv","venous"); c("lungs","pulm_vv","venous")
        c("pulm_vv","left_atrium","venous",desc="→")
        c("left_atrium","left_ventricle","cardiac",desc="")

    # ───── GI TRACT ─────
    def _build_gi(self):
        gi = ["mouth","pharynx","esophagus","stomach","duodenum","jejunum","ileum",
              "cecum","asc_colon","trans_colon","desc_colon","sig_colon","rectum","anus"]
        for i in range(len(gi)-1):
            self._c(gi[i], gi[i+1], "gi_tract", desc="")
            self._c(gi[i+1], gi[i], "gi_tract", "retrograde", desc="/")
        self._c("cecum","appendix","gi_tract","bidirectional",desc="")

    # ───── URINARY ─────
    def _build_urinary(self):
        c = self._c
        c("r_kidney","r_ureter","urinary"); c("l_kidney","l_ureter","urinary")
        c("r_ureter","bladder","urinary"); c("l_ureter","bladder","urinary")
        c("bladder","urethra","urinary")
        # retrograde (ascending infection)
        c("urethra","bladder","urinary","retrograde",desc="→")
        c("bladder","r_ureter","urinary","retrograde",desc="")
        c("bladder","l_ureter","urinary","retrograde")
        c("r_ureter","r_kidney","urinary","retrograde",desc="→")
        c("l_ureter","l_kidney","urinary","retrograde")

    # ───── BILIARY ─────
    def _build_biliary(self):
        c = self._c
        c("liver","bile_duct","biliary",desc="→")
        c("gallbladder","bile_duct","biliary","bidirectional",desc="")
        c("bile_duct","duodenum","biliary",desc="→")
        c("pancreas","duodenum","biliary",desc="→")
        c("duodenum","bile_duct","biliary","retrograde",desc="→")
        c("bile_duct","liver","biliary","retrograde",desc="→")
        c("bile_duct","gallbladder","biliary","retrograde")
        c("duodenum","pancreas","biliary","retrograde",desc="→")

    # ───── NEURAL ─────
    def _build_neural(self):
        c = self._c
        c("brain","brainstem","neural","bidirectional"); c("brainstem","spinal_cord","neural","bidirectional")
        c("brainstem","vagus_n","neural"); c("vagus_n","heart","neural",desc="→")
        c("vagus_n","lungs","neural"); c("vagus_n","esophagus","neural")
        c("vagus_n","stomach","neural",desc="→"); c("vagus_n","liver","neural")
        c("vagus_n","pancreas","neural"); c("vagus_n","jejunum","neural"); c("vagus_n","ileum","neural")
        c("spinal_cord","phrenic_n","neural",desc="C3-C5")
        c("phrenic_n","diaphragm","neural",desc="→")
        c("trachea","r_bronchus","airway"); c("trachea","l_bronchus","airway")
        c("r_bronchus","r_lung","airway"); c("l_bronchus","l_lung","airway")

    # ───── REFERRED PAIN ─────
    def _build_referred_pain(self):
        c = self._c
        c("heart","l_arm","referred_pain","bidirectional",desc="T1-T4→")
        c("heart","jaw","referred_pain","bidirectional",desc="→")
        c("heart","l_shoulder","referred_pain","bidirectional")
        c("heart","epigastrium","referred_pain","bidirectional",desc="→()")
        c("diaphragm","r_shoulder","referred_pain","bidirectional",desc="C3-C5→")
        c("gallbladder","r_shoulder","referred_pain","bidirectional",desc="→(Kehr)")
        c("liver","r_shoulder","referred_pain","bidirectional",desc="→")
        c("spleen","l_shoulder","referred_pain","bidirectional",desc="→")
        c("appendix","epigastrium","referred_pain","bidirectional",desc="→")
        c("pancreas","epigastrium","referred_pain","bidirectional",desc="→")
        c("r_kidney","r_arm","referred_pain","bidirectional",desc="")
        c("l_kidney","l_arm","referred_pain","bidirectional")
        c("uterus","lower_limbs","referred_pain","bidirectional",desc="→/")

    # ───── LYMPHATIC ─────
    def _build_lymph(self):
        c = self._c
        c("brain","cerv_ln","lymphatic"); c("thyroid","cerv_ln","lymphatic")
        c("r_lung","med_ln","lymphatic"); c("l_lung","med_ln","lymphatic"); c("lungs","med_ln","lymphatic")
        c("stomach","mes_ln","lymphatic"); c("jejunum","mes_ln","lymphatic"); c("ileum","mes_ln","lymphatic")
        c("asc_colon","mes_ln","lymphatic"); c("desc_colon","mes_ln","lymphatic")
        c("liver","med_ln","lymphatic"); c("upper_limbs","axil_ln","lymphatic")
        c("lower_limbs","ing_ln","lymphatic"); c("bladder","ing_ln","lymphatic")
        c("rectum","ing_ln","lymphatic"); c("uterus","ing_ln","lymphatic"); c("prostate","ing_ln","lymphatic")
        c("cerv_ln","thoracic_duct","lymphatic"); c("med_ln","thoracic_duct","lymphatic")
        c("mes_ln","thoracic_duct","lymphatic"); c("axil_ln","thoracic_duct","lymphatic")
        c("ing_ln","thoracic_duct","lymphatic")
        c("thoracic_duct","svc","lymphatic",desc="→()")

    # ───── ADJACENCY ─────
    def _build_adjacent(self):
        pairs = [
            ("liver","diaphragm",""), ("liver","r_kidney","Morrison"),
            ("liver","stomach",""), ("liver","gallbladder",""),
            ("liver","trans_colon",""), ("gallbladder","duodenum",""),
            ("pancreas","duodenum",""), ("pancreas","spleen",""),
            ("pancreas","stomach",""), ("pancreas","l_kidney",""),
            ("spleen","l_kidney",""), ("spleen","stomach",""),
            ("spleen","diaphragm",""), ("stomach","duodenum",""),
            ("duodenum","r_kidney",""), ("appendix","cecum",""),
            ("appendix","ileum",""), ("heart","lungs",""),
            ("heart","esophagus",""), ("heart","diaphragm",""),
            ("r_lung","pleura",""), ("l_lung","pleura",""),
            ("r_lung","diaphragm",""), ("l_lung","diaphragm",""),
            ("brain","meninges",""), ("brainstem","cerebellum",""),
            ("bladder","uterus",""), ("bladder","prostate",""),
            ("bladder","rectum",""),
            ("r_kidney","r_adrenal",""), ("l_kidney","l_adrenal",""),
            ("asc_colon","r_kidney",""), ("desc_colon","l_kidney",""),
        ]
        for s, t, d in pairs:
            self._c(s, t, "adjacent", "bidirectional", desc=d)

    def _build_blood_connections(self):
        """blood ↔ many organs. Reflects physiological reality:
        blood physically contacts every organ via vasculature.
        Each connection is 'circulates_through' (bidirectional),
        used by spread/anatomy reasoning when disease origin = blood
        (Anemia, ITP, DVT, etc.).
        """
        # Only add connections if 'blood' organ was actually defined
        # (defensive — should always be there after _build_organs).
        if "blood" not in self.organs:
            return
        blood_partners = [
            # cardiovascular axis (where blood lives)
            "heart", "aorta", "pulm_aa", "pulm_vv", "ivc", "svc",
            # systemic vessels representative samples
            "r_femoral_v", "l_femoral_v", "r_femoral_a", "l_femoral_a",
            # production / destruction sites
            "bone_marrow", "spleen", "liver",
            # filtration / endocrine
            "r_kidney", "l_kidney", "pancreas", "thyroid",
            # tissue beds
            "lungs", "brain", "skin", "bone",
        ]
        for partner in blood_partners:
            if partner in self.organs:
                self._c("blood", partner, "circulates_through", "bidirectional",
                        desc="blood-tissue exchange", w=0.6)

    # ════════════════════════════════════════════════════════════════
    #   CNS HIERARCHY (Stage 1: Central Nervous System tree)
    # ════════════════════════════════════════════════════════════════

    def _build_cns_hierarchy(self):
        """
        Central nervous system as a tree. Existing 'brain', 'brainstem',
        'cerebellum', 'spinal_cord', 'meninges' stay as parents; we add
        sub-structures (lobes, deep gray matter, spinal cord segments)
        as children.

        Coordinates: x=right(+)/left(-), y=superior(+)/inferior(-),
        z=anterior(+)/posterior(-). Brain centered around (0, 0.82, 0).

        Sources: Gray's Anatomy 41e (CNS), Kandel Principles of Neural
        Science (functional anatomy). Coordinates approximate within
        atlas convention; level field indicates depth in tree.

        All organs use system="neurologic". The hierarchy lets reasoning
        say "frontal_lobe is part of cerebrum, part of brain, part of
        nervous_system" — useful for spread paths and proof granularity.
        """

        # ── Level 0: top-level grouping (logical, no coords) ──
        # NOTE: 'brain', 'brainstem', 'cerebellum', 'spinal_cord' already exist
        # from _build_organs(). We add them parent links retroactively below.

        # ── Promote existing organs into hierarchy ──
        # brain → cerebrum (we'll add cerebrum, then move existing lobes under it)
        for nm in ["brain", "brainstem", "cerebellum", "spinal_cord", "meninges"]:
            if nm in self.organs:
                self.organs[nm].level = 1

        # ── Level 2: major brain divisions ──
        # Cerebrum (largest brain part — both hemispheres, lobes attached)
        self._oh("cerebrum",        "neurologic", "head", "midline",
                 (0, .85, 0),       ["higher_cognition", "voluntary_motor", "sensation"],
                 ["cerebral_hemispheres"], parent="brain", level=2)

        # Diencephalon — between cerebrum and brainstem
        self._oh("diencephalon",    "neurologic", "head", "midline",
                 (0, .82, -.02),    ["relay", "homeostasis"],
                 [], parent="brain", level=2)

        # ── Level 3: cerebrum sub-structures ──
        # Lobes (5 major)
        self._oh("frontal_lobe",    "neurologic", "head", "bilateral",
                 (.08, .88, .12),   ["executive_function", "voluntary_motor", "broca_speech"],
                 [], parent="cerebrum", level=3)
        self._oh("parietal_lobe",   "neurologic", "head", "bilateral",
                 (.08, .89, -.02),  ["somatosensation", "spatial_orientation"],
                 [], parent="cerebrum", level=3)
        self._oh("temporal_lobe",   "neurologic", "head", "bilateral",
                 (.14, .82, .04),   ["auditory", "memory", "wernicke_language"],
                 [], parent="cerebrum", level=3)
        self._oh("occipital_lobe",  "neurologic", "head", "bilateral",
                 (.05, .86, -.14),  ["visual_processing"],
                 [], parent="cerebrum", level=3)
        self._oh("insula",          "neurologic", "head", "bilateral",
                 (.10, .82, .02),   ["interoception", "taste"],
                 [], parent="cerebrum", level=3)

        # Deep gray matter (subcortical)
        self._oh("basal_ganglia",   "neurologic", "head", "bilateral",
                 (.05, .83, .02),   ["motor_modulation", "habit_learning"],
                 [], parent="cerebrum", level=3)
        self._oh("hippocampus",     "neurologic", "head", "bilateral",
                 (.10, .81, .02),   ["episodic_memory", "spatial_memory"],
                 [], parent="cerebrum", level=3)
        self._oh("amygdala",        "neurologic", "head", "bilateral",
                 (.08, .80, .04),   ["fear_response", "emotional_memory"],
                 [], parent="cerebrum", level=3)
        self._oh("corpus_callosum", "neurologic", "head", "midline",
                 (0, .85, 0),       ["interhemispheric_communication"],
                 [], parent="cerebrum", level=3)

        # ── Level 3: diencephalon sub-structures ──
        self._oh("thalamus",        "neurologic", "head", "bilateral",
                 (.03, .83, 0),     ["sensory_relay", "consciousness"],
                 [], parent="diencephalon", level=3)
        self._oh("hypothalamus",    "neurologic", "head", "midline",
                 (0, .81, .02),     ["homeostasis", "endocrine_control", "ans_control"],
                 [], parent="diencephalon", level=3)
        self._oh("pineal_gland",    "neurologic", "head", "midline",
                 (0, .82, -.04),    ["melatonin", "circadian_rhythm"],
                 ["pineal"], parent="diencephalon", level=3)

        # ── Level 2: brainstem sub-structures ──
        self._oh("midbrain",        "neurologic", "head", "midline",
                 (0, .79, 0),       ["eye_movement", "auditory_relay", "dopamine_origin"],
                 ["mesencephalon"], parent="brainstem", level=2)
        self._oh("pons",            "neurologic", "head", "midline",
                 (0, .76, .02),     ["respiratory_control", "cn5_7_origins"],
                 [], parent="brainstem", level=2)
        self._oh("medulla",         "neurologic", "head", "midline",
                 (0, .73, .01),     ["cardiac_respiratory_centers", "autonomic"],
                 ["medulla_oblongata"], parent="brainstem", level=2)
        self._oh("reticular_formation","neurologic","head","midline",
                 (0, .76, 0),       ["arousal", "consciousness", "sleep_wake"],
                 [], parent="brainstem", level=2)

        # ── Level 2: cerebellum sub-structures ──
        self._oh("cerebellar_vermis","neurologic","head","midline",
                 (0, .74, -.08),    ["posture", "trunk_coordination"],
                 [], parent="cerebellum", level=2)
        self._oh("cerebellar_hemispheres","neurologic","head","bilateral",
                 (.10, .74, -.08),  ["limb_coordination", "motor_learning"],
                 [], parent="cerebellum", level=2)

        # ── Level 2: spinal cord by region ──
        # Existing 'spinal_cord' covers C1-S5; we add regional segments
        # for diseases that localize (radiculopathies, cord syndromes).
        self._oh("spinal_cord_cervical","neurologic","spine","midline",
                 (0, .55, -.12),    ["upper_limb_motor_sensory", "diaphragm_innervation"],
                 [], parent="spinal_cord", level=2)
        self._oh("spinal_cord_thoracic","neurologic","spine","midline",
                 (0, .25, -.12),    ["trunk_motor_sensory", "sympathetic_outflow"],
                 [], parent="spinal_cord", level=2)
        self._oh("spinal_cord_lumbar","neurologic","spine","midline",
                 (0, -.05, -.12),   ["lower_limb_motor_sensory"],
                 [], parent="spinal_cord", level=2)
        self._oh("spinal_cord_sacral","neurologic","pelvis","midline",
                 (0, -.18, -.12),   ["pelvic_floor", "bladder_bowel_control"],
                 ["conus_medullaris"], parent="spinal_cord", level=2)

        # ── Level 2: meninges sub-layers ──
        self._oh("dura_mater",      "neurologic", "head", "midline",
                 (0, .85, .15),     ["outer_protective_layer"],
                 [], parent="meninges", level=2)
        self._oh("arachnoid_mater", "neurologic", "head", "midline",
                 (0, .85, .14),     ["csf_containment_outer"],
                 [], parent="meninges", level=2)
        self._oh("pia_mater",       "neurologic", "head", "midline",
                 (0, .85, .13),     ["inner_vascular_layer"],
                 [], parent="meninges", level=2)
        self._oh("csf_space",       "neurologic", "head", "midline",
                 (0, .82, 0),       ["csf_circulation", "cushioning"],
                 ["subarachnoid_space", "ventricles"], parent="meninges", level=2)

        # ── Hierarchical connections (parent ↔ child as "part_of") ──
        # These complement existing functional connections; they encode
        # "anatomical containment" specifically, distinct from spread paths.
        for name, organ in list(self.organs.items()):
            if organ.parent and organ.parent in self.organs:
                self._c(name, organ.parent, "part_of", "antegrade",
                        desc=f"{name} is anatomically part of {organ.parent}",
                        w=1.0)

    def _build_hierarchy_index(self):
        """Populate children[] lists by walking parent pointers.
        Runs after all _build_* methods so all organs are present."""
        for name, organ in self.organs.items():
            organ.children = []   # reset in case of re-entry
        for name, organ in self.organs.items():
            if organ.parent and organ.parent in self.organs:
                self.organs[organ.parent].children.append(name)
        # Sort children for stable iteration
        for organ in self.organs.values():
            organ.children.sort()

    # ── Hierarchy traversal helpers ──
    def ancestors(self, name: str) -> List[str]:
        """Return list of ancestors from immediate parent up to root.
        Example: ancestors('frontal_lobe') → ['cerebrum', 'brain'].
        """
        result = []
        if name not in self.organs:
            return result
        cur = self.organs[name].parent
        seen = set()
        while cur and cur not in seen:
            seen.add(cur)
            result.append(cur)
            if cur not in self.organs:
                break
            cur = self.organs[cur].parent
        return result

    def descendants(self, name: str) -> List[str]:
        """Return all descendants (children, grandchildren, ...) BFS order."""
        if name not in self.organs:
            return []
        result = []
        queue = list(self.organs[name].children)
        seen = set(queue)
        while queue:
            cur = queue.pop(0)
            result.append(cur)
            if cur in self.organs:
                for child in self.organs[cur].children:
                    if child not in seen:
                        seen.add(child)
                        queue.append(child)
        return result

    # ═══════════════════════════════════════════════
    #   API
    # ═══════════════════════════════════════════════

    def resolve(self, name: str) -> Optional[str]:
        n = (name or "").strip().lower().replace("_"," ")
        if n in self._alias: return self._alias[n]
        for k, v in self._alias.items():
            if n in k or k in n: return v
        return None

    def get_organ(self, name: str) -> Optional[Organ]:
        key = self.resolve(name)
        return self.organs.get(key) if key else None

    def get_connections_from(self, name: str, conn_types: List[str] = None) -> List[Connection]:
        key = self.resolve(name) or name
        conns = self._fwd.get(key, [])
        if conn_types:
            conns = [c for c in conns if c.conn_type in conn_types]
        return conns

    def get_neighbors(self, name: str, conn_types: List[str] = None) -> List[str]:
        return list(set(c.target for c in self.get_connections_from(name, conn_types)))

    # ═══════════════════════════════════════════════
    #   (BFS)
    # ═══════════════════════════════════════════════

    SPREAD_RULES = {
        "infection": {
            "types": ["arterial","venous","portal","gi_tract","urinary","biliary","adjacent","lymphatic","cardiac","airway"],
            "dirs": ["antegrade","retrograde","bidirectional"],
        },
        "hematogenous": {
            "types": ["arterial","venous","portal","cardiac"],
            "dirs": ["antegrade","bidirectional"],
        },
        "cancer": {
            "types": ["venous","portal","lymphatic","adjacent","arterial"],
            "dirs": ["antegrade","bidirectional"],
        },
        "embolism": {
            "types": ["venous","portal","cardiac","arterial"],
            "dirs": ["antegrade","bidirectional"],
        },
        "referred_pain": {
            "types": ["referred_pain","neural"],
            "dirs": ["antegrade","retrograde","bidirectional"],
        },
        "direct_spread": {
            "types": ["adjacent","gi_tract","biliary","urinary"],
            "dirs": ["antegrade","retrograde","bidirectional"],
        },
    }

    def find_spread_paths(
        self,
        origin: str,
        spread_type: str = "infection",
        max_hops: int = 6,
        target: str = None,
    ) -> List[Dict[str, Any]]:
        """
        BFS:  origin  spread_type 

        : [{"organ": str, "path": [str...], "hops": int, "route_types": [str...], "risk": float}]
        """
        key = self.resolve(origin)
        if not key:
            return []
        target_key = self.resolve(target) if target else None

        rules = self.SPREAD_RULES.get(spread_type, self.SPREAD_RULES["infection"])
        allowed_types = set(rules["types"])
        allowed_dirs = set(rules["dirs"])

        visited = {key}
        queue = deque([(key, [key], [], 0)])  # (node, path, route_types, hops)
        results = []

        while queue:
            node, path, rtypes, hops = queue.popleft()
            if hops >= max_hops:
                continue

            for conn in self._fwd.get(node, []):
                if conn.conn_type not in allowed_types:
                    continue
                if conn.direction not in allowed_dirs:
                    continue
                nxt = conn.target
                if nxt in visited:
                    continue

                visited.add(nxt)
                new_path = path + [nxt]
                new_rt = rtypes + [conn.conn_type]
                new_hops = hops + 1

                if nxt in self.organs and self.organs[nxt].system not in ("cardiovascular",):
                    risk = max(0.1, 1.0 - new_hops * 0.15)
                    results.append({
                        "organ": nxt,
                        "organ_cn": nxt,  # English organ name
                        "path": new_path,
                        "route_types": new_rt,
                        "hops": new_hops,
                        "risk": round(risk, 2),
                        "description": self._describe_path(new_path, new_rt),
                    })
                    if target_key and nxt == target_key:
                        return results

                queue.append((nxt, new_path, new_rt, new_hops))

        results.sort(key=lambda x: x["hops"])
        return results

    def find_path_between(self, origin: str, target: str, spread_type: str = "infection", max_hops: int = 8) -> Optional[Dict]:
        """找两个器官之间的具体路径"""
        results = self.find_spread_paths(origin, spread_type, max_hops, target=target)
        for r in results:
            if r["organ"] == (self.resolve(target) or target):
                return r
        return None

    def find_referred_pain_sources(self, pain_location: str) -> List[Dict]:
        """给定疼痛位置，找可能的内脏来源"""
        key = self.resolve(pain_location)
        if not key:
            return []
        results = []
        for conn in self._rev.get(key, []):
            if conn.conn_type == "referred_pain":
                org = self.organs.get(conn.source)
                if org:
                    results.append({
                        "source_organ": conn.source,
                        "source_cn": conn.source,  # English organ name
                        "pain_at": key,
                        "mechanism": conn.desc,
                    })
        return results

    def get_region_organs(self, region: str) -> List[Organ]:
        """获取某个区域的所有器官"""
        return [o for o in self.organs.values() if o.region == region]

    def _describe_path(self, path: List[str], rtypes: List[str]) -> str:
        parts = []
        for i in range(len(rtypes)):
            src = path[i]
            tgt = path[i+1] if i+1 < len(path) else "?"
            src_cn = self.organs[src].aliases[0] if src in self.organs and self.organs[src].aliases else src
            tgt_cn = self.organs[tgt].aliases[0] if tgt in self.organs and self.organs[tgt].aliases else tgt
            rt = rtypes[i]
            parts.append(f"{src_cn} →[{rt}]→ {tgt_cn}")
        return " → ".join(parts) if parts else ""

    # ═══════════════════════════════════════════════
    #   NEXUS 
    # ═══════════════════════════════════════════════

    def inject_into_knowledge_graph(self, kg, max_conns: int = 500):
        """把解剖学知识注入到 NEXUS 的 KnowledgeGraph"""
        count = 0
        for org in self.organs.values():
            kg.add(org.name, "is_a", "organ", 1.0, "anatomy")
            kg.add(org.name, "belongs_to_system", org.system, 1.0, "anatomy")
            kg.add(org.name, "located_in", org.region, 1.0, "anatomy")
            count += 3
        for conn in self.connections[:max_conns]:
            kg.add(conn.source, f"connects_via_{conn.conn_type}", conn.target, conn.weight, "anatomy")
            count += 1
        print(f"[ANATOMY] Injected {count} triples into knowledge graph")
        return count

    # ═══════════════════════════════════════════════
    #   JSON ( 3D )
    # ═══════════════════════════════════════════════

    def to_json(self) -> Dict:
        return {
            "organs": {
                n: {
                    "name": o.name, "system": o.system, "region": o.region,
                    "position": o.position, "pos_3d": list(o.pos_3d),
                    "functions": o.functions, "aliases": o.aliases,
                } for n, o in self.organs.items()
            },
            "connections": [
                {
                    "source": c.source, "target": c.target,
                    "type": c.conn_type, "direction": c.direction,
                    "desc": c.desc,
                } for c in self.connections
            ],
        }