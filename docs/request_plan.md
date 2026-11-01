# RICHIESTA PIANO ARCHITETTURALE ESECUTIVO (FASE 1): WHITE-BOX DIFFERENTIABLE DSP AUTOMASTERING

Ciao Argon! Giovanni (lo studente del PoliMi) ha approvato entusiasticamente il cambio di paradigma verso l'architettura **White-Box Differentiable DSP (DDSP)**.
Come richiesto dall'utente, stabiliamo ufficialmente la nostra collaborazione:
- **Argon**: Chief Architect, Planner e Senior Technical Supervisor.
- **Antigravity**: Lead Implementation Engineer ed Esecutore.

Ti chiediamo ora di redigere il **Master Plan Tecnico e la Specifica Esecutiva di Fase 1 (Twin DSP Engine: PyTorch & C++20/JUCE)**.

Inizia la tua risposta indicando lo stato nel formato standard:
STATUS: APPROVED (oppure CHANGES_REQUESTED se ritieni ci siano vincoli preliminari mancanti).

---

## CONTENUTO RICHIESTO NEL PIANO:

### 1. Repository Layout & Project Scaffolding
Definisci la struttura di directory moderna, pulita e conforme agli standard professionali audio-tech (CMake per JUCE 8, modulo C++ header-only/core separato dalla GUI, package Python per il Twin differenziabile, suite di test Catch2 + Pytest).

### 2. Specifiche Matematiche e DSP dei Moduli (Fase 1)
Definisci le equazioni formali e le implementazioni simmetriche:
- **EQ Parametrico (6 bande)**: Low-shelf, 4x Peaking, High-shelf. Specifica come calcolare i coefficienti biquad (RBJ Cookbook) e come gestire la risposta in frequenza differenziabile in PyTorch (es. tramite valutazione su griglia FFT di $H(e^{j\omega})$ vs filtri IIR differenziabili).
- **Saturatore Armonico**: Formula di soft-clipping/waveshaping differenziabile (es. approssimazione polinomiale o $\tanh$), gestione del gain e oversampling per evitare aliasing.
- **Compressore Multibanda (3 bande)**: Crossover Linkwitz-Riley del 4° ordine ($LR_4$ con fase lineare/allineata). Detector di inviluppo differenziabile (smooth envelope ballistics con attack/release logaritmici) sia in PyTorch che in C++.
- **Stereo Imager**: Matrice Mid/Side, controllo di stereo width per banda.
- **Lookahead Brickwall Limiter**: Buffer di lookahead, attack/release smoothing per il plugin C++ (e sua controparte nel training).

### 3. Protocollo di Verifica & Co-Simulazione (The Null Test Contract)
Definisci il protocollo di test automatizzato per garantire che il motore PyTorch e il motore C++ siano bit-accurate a parità di parametri $\theta$ (target: Null Test $< -60\text{ dBFS}$ o MSE $< 10^{-6}$ su segnali di test deterministici).

### 4. Roadmap Dettagliata di Fase 1 (Ticket Esecutivi per Antigravity)
Suddividi la Fase 1 in step sequenziali precisi e azionabili (Task 1.1, Task 1.2, ecc.) che io (Antigravity) implementerò e ti sottoporrò per revisione e collaudo.
