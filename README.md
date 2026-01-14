# Trading-Bot-Binance-BTCUSDC
Trading Bot Binance 
# 🤖 BTCUSDC 15m Trading Bot (Binance)

Bot de trading **algorithmique professionnel** développé en **Python**, conçu pour trader la paire **BTC/USDC** sur **Binance** en **timeframe 15 minutes**.

Le bot repose sur une **stratégie avancée basée sur les indicateurs TEMA 20 / TEMA 50**, avec analyse de pente, accélération, momentum et gestion **multi-positions cycliques**. Il fonctionne en **simulation (paper trading)** avec reporting en temps réel via **Telegram**.

> ⚠️ **Avertissement** : Ce projet est à but éducatif et expérimental. Il ne constitue en aucun cas un conseil financier.

---

## ✨ Fonctionnalités principales

* 📈 **Stratégie TEMA 20 / TEMA 50** (15 minutes)
* 🔁 **Gestion multi-positions en cycles continus** (Achat → Vente → Achat)
* 🧠 Détection de :

  * Croisements haussiers / baissiers
  * Débuts de tendance (LONG / SHORT)
  * Creux & sommets (rebonds / retournements)
  * Accélération et momentum
* 💾 **Stockage local SQLite** (OHLC + indicateurs)
* 📊 Calcul automatique des indicateurs techniques
* 💬 **Notifications Telegram en temps réel**
* 💰 **Wallet simulé** (USDC / BTC)
* ⚙️ Architecture modulaire et lisible

---

## 🧠 Logique de trading (résumé)

### 📊 Indicateurs utilisés

* **TEMA 20** : détection court terme
* **TEMA 50** : tendance principale

### 📌 Signaux BUY

* Croisement haussier TEMA20 > TEMA50 avec pente et accélération positives
* Rebond sur creux pendant une phase LONG

### 📌 Signaux SELL

* Croisement baissier TEMA20 < TEMA50
* Sommet détecté en phase SHORT
* Vente uniquement si le **seuil de profit** est atteint

### 🔁 Gestion des positions

* Positions fractionnées (ex: `0.0001 BTC` par trade)
* Chaque position suit un cycle :

  * `WAIT_BUY` → `WAIT_SELL` → `WAIT_BUY`
* Possibilité de **plusieurs positions actives simultanément**
* Le bot calcule automatiquement :

  * Positions achetables
  * Capital engagé
  * Capital libre

---

## 🏗️ Architecture du projet

```
📦 trading-bot-btcusdc
 ┣ 📜 bot.py                # Script principal
 ┣ 📜 db15M.db              # Base de données SQLite
 ┣ 📜 README.md             # Documentation
```

### 🗄️ Base de données (SQLite)

Table `ohlc` :

* Données de marché : OHLCV
* Indicateurs : TEMA20, TEMA50, RSI (prévu), etc.

---

## ⚙️ Configuration

### 🔐 Variables importantes

```python
SYMBOL = "BTCUSDC"
INTERVAL = "15m"
wallet_usdc = 100.0
qtt_trade_btc = 0.0001
seuil = 0.03  # 3% de profit
```

### 🔔 Telegram

* Bot créé via **@BotFather**
* Notifications :

  * Démarrage / arrêt
  * Nouvelles bougies
  * Signaux BUY / SELL
  * État du wallet

---

## ▶️ Lancement du bot

```bash
python bot.py
```

Le bot :

1. Initialise la base de données
2. Télécharge l’historique des bougies
3. Calcule les indicateurs
4. Attend chaque clôture de bougie 15m
5. Analyse → Décide → Simule les ordres

---

## 📊 Exemple de message Telegram

```
📊 ÉTAT: LONG | Prix: 65432.50
📏 Écart TEMA20/50: +0.21%
📐 Pente TEMA20: +34.2
⚡ Accélération: +12.6
🟢 SIGNAL BUY détecté
```

---

## 🧪 Mode actuel

* ❌ Trading réel désactivé
* ✅ Simulation complète (paper trading)
* ✅ Prêt pour backtesting / amélioration

---

## 🚧 Améliorations prévues

* [ ] Backtesting automatique
* [ ] Mode réel Binance Spot
* [ ] Gestion du risque (SL / TP dynamiques)
* [ ] Dashboard graphique
* [ ] Multi-symboles
* [ ] Optimisation des paramètres

---

## 📚 Dépendances

```txt
python-binance
pandas
numpy
matplotlib
python-telegram-bot
sqlite3
```

---

## 🧠 Philosophie du projet

Ce bot est conçu comme un **framework de recherche et d’expérimentation** autour :

* de la **structure de marché**
* des **cycles de tendance**
* de la **gestion fine des positions**

Il privilégie la **lisibilité**, la **traçabilité** et la **robustesse** plutôt que l’over-optimisation.

---

## ⚠️ Disclaimer

Le trading comporte des risques importants. L’auteur ne pourra être tenu responsable des pertes financières. Utilisez ce bot à vos propres risques.

---

## 👤 Auteur

**Léo De Clercq**
Bot de trading Python – Binance BTCUSDC

---

⭐ Si ce projet vous aide, n’hésitez pas à lui laisser une étoile sur GitHub !

