
# KBQA demo
 
Download wget lang_title2wikidataID-normalized_with_redirect.pkl file for fast label to entity ID to telegram bot directory
```bash
wget https://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl
mv lang_title2wikidataID-normalized_with_redirect.pkl ./data/
```

```bash
export TELEGRAM_BOT_KEY=$TELEGRAM_BOT_KEY
docker-combopse build && docker-combopse up
```