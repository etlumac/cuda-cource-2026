#!/usr/bin/env bash
# Проверки командной строки и формата файлов.
# Запуск из каталога лабораторной: bash tests/run_checks.sh [путь к программе]
BIN="${1:-./build/submission}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAILED=0

# expect_code <ожидаемый код> <описание> <команда...>
expect_code() {
  local want="$1"
  local name="$2"
  shift 2
  "$@" >"$TMP/stdout.txt" 2>"$TMP/stderr.txt"
  local got=$?
  if [ "$got" -eq "$want" ]; then
    echo "OK    $name (код $got)"
  else
    echo "FAIL  $name: ожидался код $want, получен $got"
    FAILED=1
  fi
}

# expect_same <описание> <файл1> <файл2>
expect_same() {
  if diff -q "$2" "$3" >/dev/null; then
    echo "OK    $1"
  else
    echo "FAIL  $1: файлы различаются"
    FAILED=1
  fi
}

# make_input <M> <K> <N> <файл>: случайные числа, seed задан постоянным.
make_input() {
  awk -v m="$1" -v k="$2" -v n="$3" 'BEGIN {
    srand(2026);
    print m, k, n;
    for (i = 0; i < m * k + k * n; i++) printf "%.4f\n", rand() * 20 - 10;
  }' >"$4"
}

# check_shape <M> <N> <файл>: заголовок "M N" и ровно M строк по N чисел.
check_shape() {
  local header rows bad
  header="$(head -n 1 "$3")"
  rows="$(($(wc -l <"$3") - 1))"
  bad="$(tail -n +2 "$3" | awk -v n="$2" 'NF != n' | wc -l)"
  if [ "$header" = "$1 $2" ] && [ "$rows" -eq "$1" ] && [ "$bad" -eq 0 ]; then
    echo "OK    форма результата $1 x $2"
  else
    echo "FAIL  форма результата $1 x $2"
    FAILED=1
  fi
}

# Пример из задания, оба режима.
expect_code 0 "пример, naive" "$BIN" --input data/lab2.txt --output "$TMP/naive.txt" --mode naive
expect_same "пример, naive совпал с ожидаемым" "$TMP/naive.txt" data/lab2_expected.txt
expect_code 0 "пример, tiled (по умолчанию)" "$BIN" --input data/lab2.txt --output "$TMP/tiled.txt"
expect_same "пример, tiled совпал с ожидаемым" "$TMP/tiled.txt" data/lab2_expected.txt

# Форма результата для разных размеров, в том числе не кратных тайлу.
for size in "1 1 1" "3 5 2" "17 33 15" "127 65 257"; do
  set -- $size
  make_input "$1" "$2" "$3" "$TMP/in.txt"
  for mode in naive tiled; do
    expect_code 0 "размер $1x$2x$3, $mode" "$BIN" --input "$TMP/in.txt" --output "$TMP/out.txt" --mode "$mode"
    check_shape "$1" "$3" "$TMP/out.txt"
  done
done

# Ошибки.
expect_code 0 "--help" "$BIN" --help
expect_code 2 "запуск без аргументов" "$BIN"
expect_code 2 "нет --output" "$BIN" --input data/lab2.txt
expect_code 2 "нет --input" "$BIN" --output "$TMP/out.txt"
expect_code 2 "несуществующий входной файл" "$BIN" --input "$TMP/none.txt" --output "$TMP/out.txt"
expect_code 2 "--mode wrong" "$BIN" --input data/lab2.txt --output "$TMP/out.txt" --mode wrong
expect_code 2 "неизвестный аргумент --foo" "$BIN" --input data/lab2.txt --output "$TMP/out.txt" --foo
expect_code 2 "--mode без значения" "$BIN" --input data/lab2.txt --output "$TMP/out.txt" --mode

printf '2 3 2\n1 2 3\n4 5 6\n7 8\n9 10\n' >"$TMP/short.txt"
expect_code 2 "в файле не хватает чисел" "$BIN" --input "$TMP/short.txt" --output "$TMP/out.txt"
printf '0 3 2\n' >"$TMP/zero_m.txt"
expect_code 2 "M = 0" "$BIN" --input "$TMP/zero_m.txt" --output "$TMP/out.txt"
printf '2 0 2\n' >"$TMP/zero_k.txt"
expect_code 2 "K = 0" "$BIN" --input "$TMP/zero_k.txt" --output "$TMP/out.txt"
printf '2 3 -1\n' >"$TMP/neg.txt"
expect_code 2 "отрицательный размер" "$BIN" --input "$TMP/neg.txt" --output "$TMP/out.txt"
printf 'abc\n' >"$TMP/text.txt"
expect_code 2 "в файле текст вместо чисел" "$BIN" --input "$TMP/text.txt" --output "$TMP/out.txt"
expect_code 1 "выходной файл в несуществующей папке" "$BIN" --input data/lab2.txt --output "$TMP/no_dir/out.txt"

if [ "$FAILED" -eq 0 ]; then
  echo "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ"
else
  echo "ЕСТЬ ПРОВАЛЕННЫЕ ПРОВЕРКИ"
  exit 1
fi
