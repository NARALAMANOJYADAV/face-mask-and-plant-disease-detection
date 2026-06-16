# Deploying to Render — Step by Step

## Prerequisites
- A [Render](https://render.com) account (free tier works)
- This repo pushed to GitHub: https://github.com/NARALAMANOJYADAV/face-mask-and-plant-disease-detection
- A Groq API key from https://console.groq.com

---

## Step 1 — Connect GitHub to Render

1. Go to https://render.com and **sign in** (or sign up with your GitHub account).
2. On the Render dashboard click **"New +"** → **"Web Service"**.
3. Click **"Connect account"** under GitHub if not already connected, and authorize Render.

---

## Step 2 — Select Your Repository

1. Search for `face-mask-and-plant-disease-detection` in the repo list.
2. Click **Connect** next to it.

---

## Step 3 — Configure the Web Service

Fill in these fields:

| Field | Value |
|-------|-------|
| **Name** | `face-mask-plant-disease` (or any name you like) |
| **Region** | Choose the closest to you |
| **Branch** | `master` |
| **Root Directory** | `project` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free |

---

## Step 4 — Add Environment Variables

Scroll down to the **Environment Variables** section and add:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | your Groq API key |
| `DATABASE_URL` | `postgresql://detection_zcmj_user:CLIZ4GFYktjBTJKAMy9LhDIRD6AXbppa@dpg-d8oh0jsm0tmc73cof6fg-a/detection_zcmj` |

> Use the **Internal Database URL** (no `.oregon-postgres.render.com` in the hostname) — it's faster and works only between Render services. Use the External URL only for local development.

---

## Step 5 — Deploy

1. Click **"Create Web Service"**.
2. Render will:
   - Clone your repo
   - Run `pip install -r requirements.txt`
   - Start the uvicorn server
3. Watch the **Logs** tab — deployment takes ~2–4 minutes.
4. Once you see:
   ```
   INFO:     Application startup complete.
   ```
   your app is live.

---

## Step 6 — Open Your App

Render gives you a URL like:
```
https://face-mask-plant-disease.onrender.com
```

Your pages will be at:
- **Home:** `https://your-app.onrender.com/ui/`
- **Scan:** `https://your-app.onrender.com/ui/upload.html`
- **Admin:** `https://your-app.onrender.com/ui/admin.html`

---

## Notes on Free Tier

| Limitation | Detail |
|-----------|--------|
| **Sleep after 15 min** | Free services spin down when idle. First request after sleep takes ~30 sec to wake up. |
| **No persistent disk** | SQLite DB is stored in `/tmp` and resets on each deploy/restart. Scan history will be lost. |
| **512 MB RAM** | TensorFlow is installed but the `.h5` model is not used (Groq handles predictions). If you add the model file, you may hit memory limits. |

---

## Redeploying After Code Changes

Every `git push` to `master` will **automatically trigger a new deploy** on Render (auto-deploy is on by default).

```bash
git add .
git commit -m "your message"
git push origin master
```

---

## Troubleshooting

**"Application failed to start"**
- Check the Logs tab in Render dashboard for the exact error.
- Most common cause: missing environment variable. Make sure `GROQ_API_KEY` and `RENDER=true` are set.

**Predictions not working**
- Check that `GROQ_API_KEY` is set correctly in the Environment Variables section.
- Test the key at https://console.groq.com/playground first.

**Static files not found (404)**
- Make sure **Root Directory** is set to `project` in Render settings.
