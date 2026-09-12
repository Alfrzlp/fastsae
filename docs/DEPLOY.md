# Setup & Deploy — ridsonap.github.io/fastsae

## One-time Setup

```bash
git clone https://github.com/ridsonap/ridsonap.github.io.git ~/ridsonap.github.io
mkdir -p ~/ridsonap.github.io/fastsae
```

## Deploy (each update)

```bash
cp site/index.html ~/ridsonap.github.io/fastsae/
cd ~/ridsonap.github.io
git add fastsae/
git commit -m "Update fastsae docs"
git push origin main
```

Or:
```bash
bash site/deploy.sh
```

Site live at: https://ridsonap.github.io/fastsae
