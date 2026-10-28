#!/bin/bash
# Entrena LineVul desde microsoft/codebert-base sobre FFmpeg+QEMU y lo prueba al final.
#
# Base: README.md de LineVul, lineas 222-241 ("To retrain the RQ1 model").
# Diferencias con el comando del README (todas justificadas en
# LineVul-main/DOCUMENTACION_ENTORNO_LINEVUL.md, seccion 13):
#   --train_batch_size 4 + --gradient_accumulation_steps 4   lote efectivo 16 como el paper;
#                                                            batch 16 no cabe en 8 GB
#   rutas /datos/...                                         conjuntos_de_datos montado con un 2.o -v
#   --test_data_file ...test_linevul.csv                     test con flaw_line y flaw_line_index vacias
#   --output_dir y --model_name propios                      no mezclar con el checkpoint de los autores
#   --write_raw_preds                                        predicciones por funcion, para alinear con AMPLE
#
# Requiere el fix #1 en linevul_main.py (max_steps y save_steps entre la acumulacion).
# Tarda del orden de 5.5 h (ESTIMACION, ver DOCUMENTACION seccion 13.2). No hay reanudacion.
#
# Uso, desde la raiz del repositorio:   bash protocolo_comparacion/entrenar_linevul_ffmpeg_qemu.sh
#Establecemos que si algún comando del script falla el script debe finalizar y no continuar
set -e

#Es la ruta absoluta en donde esta el script como tal
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

#Es la ruta en donde está el conjunto de datos con el cual se entra a linevul en este caso entrenamos con ffmpeg + qemu
SPLIT="$RAIZ/conjuntos_de_datos/ffmpeg_qemu/split_semilla42"

#Esta es la ruta al archivo de registro, en donde se incluye el tema de la fecha y hora
LOG="$RAIZ/LineVul-main/entrenamiento_ffmpeg_qemu_$(date +%F_%H%M).log"

# --- comprobaciones previas -------------------------------------------------
#lo que se hace aca es verificar es que dichos archivos existan en el directorio y pues si es que uno de ellos no esta el script termina automáticamente
for f in train.csv val.csv test_linevul.csv; do
    [[ -f "$SPLIT/$f" ]] || { echo "Falta $SPLIT/$f"; exit 1; }
done

# Se hace la verificación de que las líneas que se ven en el comando que tenemos en los archivos están comentadas, si no es asi
# el script termina automáticamente
[[ $(grep -c "^    args\.\(max\|save\)_steps = .*// args\.gradient_accumulation_steps" \
    "$RAIZ/LineVul-main/linevul/linevul_main.py") -eq 2 ]] \
    || { echo "No se ven las 2 lineas del fix #1 (max_steps y save_steps) en linevul_main.py"; exit 1; }

# lo que hacen estos comandos es verificar cuanto esta que se usa de memoria en la gpu si pasa de 1500 mb, dara error el script con error
USADA=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [[ "$USADA" -gt 1500 ]]; then
    echo "La GPU ya tiene $USADA MiB en uso: cierra otros programas (el entrenamiento llega a ~5.5 GB de 8 GB)."
    exit 1
fi

echo "Log: $LOG"
cd "$RAIZ/LineVul-main"

docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v "$(pwd)":/app \
    -v "$RAIZ/conjuntos_de_datos":/datos:ro \
    linevul:cu128 python linevul_main.py \
    --output_dir=./saved_models_ffmpeg_qemu \
    --model_name=linevul_ffmpeg_qemu.bin \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_test \
    --train_data_file=/datos/ffmpeg_qemu/split_semilla42/train.csv \
    --eval_data_file=/datos/ffmpeg_qemu/split_semilla42/val.csv \
    --test_data_file=/datos/ffmpeg_qemu/split_semilla42/test_linevul.csv \
    --epochs 10 \
    --block_size 512 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --write_raw_preds \
    --seed 123456 \
    > "$LOG" 2>&1

echo "Terminado. Resultados al final de: $LOG"
echo "Renombra LineVul-main/linevul/results/raw_preds.csv a raw_preds_ffmpeg_qemu.csv antes de otra corrida."
