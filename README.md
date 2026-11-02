# 🫀 hubertECG

Application d'**analyse d'ECG en médecine générale, 100 % locale sur le téléphone**.
Aucun serveur, aucune API cloud : le LLM tourne **directement dans le navigateur du téléphone**
grâce à [WebLLM](https://github.com/mlc-ai/web-llm) (WebGPU). **Aucune donnée patient ne quitte
l'appareil.** Une fois le modèle téléchargé une première fois, l'app fonctionne **hors-ligne**.

L'app produit un **rapport structuré** : analyse du tracé, **diagnostic rapporté**, diagnostics
différentiels, **explication pédagogique** et conduite à tenir.

Deux modèles IA embarqués, **100 % locaux** :

1. **HuBERT-ECG** (foundation model ECG, Coppola et al. — pré-entraîné sur 9,1 M d'ECG 12
   dérivations, fine-tuné sur Cardio-Learning) : analyse le **signal brut** (CSV 12 dérivations)
   et détecte **164 pathologies** avec probabilité (AUROC publiés 0.84–0.99). Tourné en ONNX
   via ONNX Runtime Web (WASM).
2. Un **petit LLM** (WebLLM/WebGPU) : rédige le **rapport clinique** (diagnostic rapporté,
   explication, conduite à tenir) à partir des mesures, du contexte patient et des détections
   HuBERT-ECG.

## Fonctionnement

```
hubertECG/  (PWA statique, aucun backend)
├── index.html            # Interface mobile installable
├── app.js                # Orchestration : HuBERT-ECG + LLM + rapport
├── ecg-model.js          # Prétraitement ECG + inférence ONNX HuBERT-ECG
├── style.css
├── sw.js                 # Service worker : mode hors-ligne après 1er chargement
├── manifest.webmanifest  # "Ajouter à l'écran d'accueil"
├── models/
│   ├── hubert-ecg-small.onnx   # HuBERT-ECG SFT small exporté en ONNX (~800 Ko)
│   └── labels.json             # Les 164 labels Cardio-Learning
├── scripts/export_hubert_ecg.py # Re-génère l'export ONNX depuis Hugging Face
├── vendor/
│   ├── web-llm.js        # WebLLM embarqué
│   └── ort/              # ONNX Runtime Web + WASM embarqués
└── icons/                # Icônes de l'app
```

## Utilisation sur le téléphone

1. **Hébergez** ce dossier sur n'importe quel hébergement statique (GitHub Pages, Netlify,
   Cloudflare Pages) — l'URL doit être en **HTTPS** (requis par WebGPU et les service workers).
2. Ouvrez l'URL dans **Chrome/Edge (Android)** ou **Safari (iPhone, iOS 17+ avec WebGPU)**.
3. Menu → **« Ajouter à l'écran d'accueil »** : l'app s'installe comme une vraie application.
4. Choisissez un modèle (SmolLM2 360M recommandé, ~250 Mo), cliquez **« Charger le modèle »**
   (téléchargement unique, mis en cache par le navigateur).
5. Renseignez le contexte patient, prenez une **photo du tracé ECG** (elle reste sur l'appareil)
   et/ou les mesures (FC, PR, QRS, QT/QTc, ST…).
6. Importez un **CSV du signal 12 dérivations** (12 colonnes, indiquez la fréquence
   d'échantillonnage) : la fréquence cardiaque est calculée localement, puis **HuBERT-ECG**
   analyse le signal et retourne les pathologies détectées avec leur probabilité.
7. **« Lancer l'analyse locale »** → le LLM rédige le rapport structuré (diagnostic rapporté,
   explication, conduite à tenir) en s'appuyant sur les détections HuBERT-ECG.

## Modèles embarqués (au choix, tous quantifiés q4, adaptés au mobile)

| Modèle | Taille approx. | Usage |
|---|---|---|
| SmolLM2 360M Instruct q4 | ~250 Mo | recommandé téléphone |
| SmolLM2 135M Instruct q4 | ~120 Mo | appareils anciens |
| Qwen2.5 0.5B Instruct q4 | ~350 Mo | bon compromis |
| Llama 3.2 1B Instruct q4 | ~600 Mo | plus lent, plus capable |

Les poids sont récupérés une fois depuis le CDN Hugging Face, puis mis en cache par le
navigateur : ensuite, l'app marche sans aucune connexion.

## Développement local

```bash
python3 -m http.server 8000
# ouvrir http://localhost:8000 (WebGPU nécessite HTTPS hors localhost)
```

## Prétraitement du signal (conforme au papier HuBERT-ECG)

Filtre passe-bande 0,05–47 Hz à la fréquence d'origine → resampling à 100 Hz → fenêtre centrale
de 5 s (12 × 500 échantillons) → normalisation min-max [-1, 1] par dérivation → entrée
`[1, 6000]` du modèle ONNX → sigmoïde sur les 164 logits.

## ⚠️ Avertissement médical

Cet outil est une **aide à la décision** générée localement par IA. Il ne remplace pas le
jugement clinique du médecin ni l'avis d'un cardiologue. Toute suspicion de syndrome coronarien
aigu ou d'arythmie grave impose un **avis médical immédiat / les urgences**. Les petits modèles
embarqués peuvent se tromper : toujours confronter au contexte clinique.
