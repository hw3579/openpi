set -euo pipefail
FILE="/home/jiaqi/openpi/logs/client_0.jsonl"
BACKUP="${FILE}.bak_$(date +%s)"
cp -p "$FILE" "$BACKUP"
sed -E '/"round"[[:space:]]*:[[:space:]]*17([[:space:]]*,|[[:space:]]*})/d' "$FILE" > "${FILE}.tmp"
mv "${FILE}.tmp" "$FILE"
echo "Backup: $BACKUP" && echo "Filtered: $FILE"