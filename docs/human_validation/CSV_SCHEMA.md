# Schema dei File CSV per le Sessioni di Ascolto Umano

Questo documento definisce le colonne dei template CSV utilizzati per registrare i risultati degli esperimenti di validazione umana (`SPRINT1_LISTENING_TEMPLATE.csv`, `SPRINT2_LISTENING_TEMPLATE.csv` e i fogli `*_listening_sheet.csv` generati automaticamente da `make_blind_set.py`).

---

## 1. Scala di Valutazione Standard (1–5)

Per tutti i campi con suffisso `_1_5`:
- `1` = Decisamente peggiore / problema evidente
- `2` = Leggermente peggiore
- `3` = Neutro / equivalente / trasparente
- `4` = Leggermente migliore
- `5` = Decisamente migliore
- `NV` = Non valutabile (se il brano o la sezione non contiene l'elemento, es. assenza di voce per `vocal_clarity_1_5`)

Per il campo `distortion_artifacts_1_5`:
- `5` = Nessun artefatto udibile (pulito)
- `3` = Lieve colorazione accettabile
- `1` = Distorsione, click, dropout o pumping evidente e fastidioso

---

## 2. Schema `SPRINT1_LISTENING_TEMPLATE.csv` (Esperimento A)

| Colonna | Tipo | Descrizione |
|---|---|---|
| `track_id` | string | Identificativo anonimo del brano (es. `track01`). |
| `session_date` | string | Data e ora della sessione (`YYYY-MM-DD HH:MM`). |
| `listener` | string | Nome dell'ascoltatore. |
| `monitoring_system` | string | Cuffie o monitor + interfaccia audio utilizzati. |
| `monitoring_level_note` | string | Livello di ascolto indicativo. |
| `legacy_mode` | string | Policy usata per la condizione B (`strict`, `clamped_to_current_bounds`, `raw_theta_on_current_engine`, `current_z_half`). |
| `same_input_confirmed` | `yes` \| `no` | Conferma che A, B e C derivano dallo stesso file trimmato. |
| `alignment_confirmed` | `yes` \| `no` | Conferma che durata e allineamento (159 campioni compensati) coincidono. |
| `listening_condition` | `loudness_matched` \| `program_level` | Indica se la riga si riferisce all'ascolto cieco pareggiato in loudness o a livello reale di programma. |
| `anonymous_label` | string | Etichetta ascoltata (`blind_A`, `blind_B`, `blind_C` oppure `A_true_bypass`, ecc.). |
| `tonal_balance_1_5` | `1..5` \| `NV` | Valutazione bilanciamento tonale. |
| `bass_control_1_5` | `1..5` \| `NV` | Valutazione controllo basse frequenze. |
| `vocal_clarity_1_5` | `1..5` \| `NV` | Valutazione chiarezza voce / strumento principale. |
| `transient_punch_1_5` | `1..5` \| `NV` | Valutazione tenuta dei transienti. |
| `stereo_image_1_5` | `1..5` \| `NV` | Valutazione immagine stereo e compatibilità mono. |
| `dynamic_naturalness_1_5` | `1..5` \| `NV` | Valutazione naturalezza dinamica. |
| `distortion_artifacts_1_5` | `1..5` \| `NV` | Assenza di distorsioni/artefatti (`5` = nessun artefatto). |
| `overall_preference_1_5` | `1..5` \| `NV` | Giudizio globale sulla variante. |
| `difference_audibility` | `none` \| `subtle` \| `clear` | Udibilità della differenza rispetto alle altre varianti. |
| `preference_strength_1_5` | `1..5` | Intensità della preferenza espressa. |
| `confidence_1_5` | `1..5` | Fiducia dell'ascoltatore nella propria valutazione. |
| `clicks_dropouts_pumping_notes` | string | Segnalazione specifica di click, dropout, pumping o alterazioni stereo. |
| `free_notes` | string | Osservazioni libere. |
| `unblinded_ground_truth` | string | Compilato **dopo** il test aprendo `track01_blind_key.json` (`A_true_bypass`, `B_legacy_baseline`, `C_corrected_neutral`). |

---

## 3. Schema `SPRINT2_LISTENING_TEMPLATE.csv` (Esperimento B)

Una riga per ogni combinazione `track × section × level_matched` (minimo `3 brani × 3 sezioni × 2 condizioni di livello = 18 righe`):

| Colonna | Tipo | Descrizione |
|---|---|---|
| `track_id` | string | ID anonimo del brano (`track01`, `track02`, `track03`). |
| `section_id` | string | ID della sezione (`section01`, `section02`, `section03`). |
| `section_role` | string | Ruolo musicale della sezione (`quiet_intro_or_verse`, `dense_chorus_or_drop`, `transient_bridge_or_outro`). |
| `start_sec` | float | Timestamp iniziale della sezione nel brano completo (in secondi). |
| `end_sec` | float | Timestamp finale della sezione nel brano completo (in secondi). |
| `reference_mode` | `same_track` \| `stylistic` | Modalità di riferimento utilizzata in ottimizzazione. |
| `level_matched` | `yes` \| `no` | `yes` = copia con guadagno statico pareggiato (`<= -1.0 dBTP`); `no` = livello di programma effettivo. |
| `variant_a_anon_id` | string | Nome del primo file anonimo (es. `blind_A`). |
| `variant_b_anon_id` | string | Nome del secondo file anonimo (es. `blind_B`). |
| `choice_A_B_ND` | `A` \| `B` \| `ND` | Variante preferita (`A`, `B`, oppure `ND` = nessuna differenza percepibile). |
| `preference_strength_1_5` | `1..5` | Intensità della preferenza tra A e B. |
| `confidence_1_5` | `1..5` | Fiducia nella valutazione. |
| `tonal_balance_1_5` .. `overall_preference_1_5` | `1..5` \| `NV` | Valutazioni per criterio (riferite alla variante scelta o a `blind_B` rispetto a `blind_A`). |
| `artifacts_noted` | string | Eventuali artefatti uditi (pumping, saturazione, sibilanti, perdita punch). |
| `free_notes` | string | Note musicali libere sulla sezione. |
| `unblinded_variant_a` | string | Compilato **dopo** l'ascolto aprendo `*_blind_key.json`. |
| `unblinded_variant_b` | string | Compilato **dopo** l'ascolto aprendo `*_blind_key.json`. |
