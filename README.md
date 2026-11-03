# 🫀 hubertECG

Application **Android native** d'analyse d'ECG en médecine générale, **100 % on-device** :
aucun serveur, aucune donnée patient ne quitte le téléphone.

Deux modèles IA embarqués :

1. **HuBERT-ECG** (foundation model ECG, Coppola et al. — pré-entraîné sur 9,1 M d'ECG 12
   dérivations, fine-tuné sur Cardio-Learning) : analyse le **signal brut** (CSV 12 dérivations)
   et détecte **164 pathologies** avec probabilité (AUROC publiés 0.84–0.99). Tourné en ONNX
   via **ONNX Runtime Android**.
2. Un **LLM embarqué** via **MediaPipe LLM Inference** (GPU/NNAPI) : rédige le **rapport
   clinique** (diagnostic rapporté, explication, conduite à tenir, niveau d'urgence) à partir
   des mesures, du contexte patient et des détections HuBERT-ECG. Le modèle `.task` est téléchargé
   **une seule fois** puis mis en cache : l'app fonctionne ensuite **hors-ligne**.

## Structure

```
android/                        # Projet Android (Gradle, Kotlin)
├── app/src/main/java/com/hubertecg/
│   ├── MainActivity.kt         # UI : patient, photo, CSV, orchestration
│   ├── EcgAnalyzer.kt          # Prétraitement ECG + inférence ONNX HuBERT-ECG
│   └── LlmClient.kt            # LLM MediaPipe + prompt + parsing du rapport JSON
├── app/src/main/assets/models/ # hubert-ecg-small.onnx + labels.json (embarqués)
└── app/src/main/res/           # layout, manifest, thèmes
models/                         # Modèle ONNX + labels (source, aussi copiés dans assets)
scripts/export_hubert_ecg.py    # Re-génère l'export ONNX depuis Hugging Face
index.html / app.js / …         # Ancienne version PWA (conservée)
```

## Build

```bash
cd android
./gradlew assembleDebug        # ou assembleRelease
# APK : app/build/outputs/apk/debug/app-debug.apk
```

Pré-requis : JDK 17 + Android SDK (compileSdk 35, minSdk 26). Les dépendances (ONNX Runtime
Android, MediaPipe tasks-genai, CameraX) sont tirées de Google Maven / Maven Central.

## Utilisation

1. Renseignez le contexte patient (âge, sexe, antécédents, symptômes, traitements).
2. Prenez une **photo du tracé ECG** (restée dans le cache interne de l'app).
3. Importez un **CSV du signal 12 dérivations** (12 colonnes, indiquez la fréquence en Hz) :
   fréquence cardiaque calculée localement, puis **HuBERT-ECG** retourne les 164 pathologies
   détectées avec leur probabilité.
4. **« Rapport clinique (LLM local) »** : le LLM embarqué rédige le rapport structuré.

## Prétraitement du signal (conforme au papier HuBERT-ECG)

Filtre passe-bande 0,05–47 Hz à la fréquence d'origine → resampling 100 Hz → fenêtre centrale
de 5 s (12 × 500 échantillons) → min-max [-1, 1] par dérivation → entrée `[1, 6000]` ONNX →
sigmoïde sur les 164 logits.

## ⚠️ Avertissement médical

Aide à la décision par IA embarquée. Ne remplace ni le jugement clinique du médecin ni l'avis
d'un cardiologue. Toute suspicion de syndrome coronarien aigu ou d'arythmie grave impose un
**avis médical immédiat / les urgences**.
