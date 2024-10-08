# Симулация с клетъчни автомати на кристален растеж

## Имплементирани възможности

За момента симулацията поддържа:

- [x] Безусловно A2A правило (агрегация) в 2D (Вероятност 1-pA2k *)
- [x] Свързване (attach-to-kink) в 2D (Вероятност pA2k)
- [x] Множество стъпки на дифузия (nds)

* **Вж. аргументите на командния ред за обяснение на pA2k**

## Инсталация

За да инсталирате необходимите зависимости, създайте виртуална среда (по желание, но се препоръчва) и изпълнете:

```bash
pip install -r requirements.txt
```

## Използване

За да видите наличните опции на командния ред, може да извикате скрипта с флага --help:

```bash
python ca.py --help
```
Това ще покаже съобщение за помощ с подробности за всеки аргумент.

## Аргументи на командния ред

Скриптът приема следните аргументи на командния ред:

- `-N`:
  - Тип: int
  - По подразбиране: 100
  - Описание: Размер на симулационната решетка.

- `-D`:
  - Тип: float
  - По подразбиране: 0.1
  - Описание: Началната част от клетките, които дифундират.

- `-T`:
  - Тип: int
  - По подразбиране: 100
  - Описание: Брой времеви стъпки за изпълнение на симулацията.

- `--nds`:
  - Тип: int
  - По подразбиране: 5
  - Описание: Брой дифузионни стъпки във всяка времева стъпка.

- `--pA2k`:
  - Тип: float
  - По подразбиране: 0.8
  - Описание: Вероятност за прилагане на правилото A2K (attach-to-kink).

- `--save-dir`:
  - Тип: str
  - По подразбиране: ca_output
  - Описание: Папка, в която ще се запазят кадрите от симулацията.

- `--save-npz`:
  - Тип: bool
  - По подразбиране: False
  - Описание: Ако е зададено, ще запази състоянието като .npz файлове.

- `--save-plots`:
  - Тип: bool
  - По подразбиране: True
  - Описание: Ако е зададено, ще запази графики на състоянието при всяка стъпка.

## Примерен старт с аргументи:
```bash
python3 ./ca.py -T 1000 -D 0.1 -N 500 --nds 5 --pA2k 0.8 --save-dir ca_output
```

Това ще стартира симулацията с решетка 500x500, начална концентрация от 10% дифузионни клетки, 5 дифузионни стъпки, 80% вероятност за свързване към кинк и ще запази резултатите в папката ca_output.

## Обяснение на действието на pA2k

Списъкът с дифундиращи частици се разделя на два под-списъка: с агрегриращи и свърващи се към кинк. Вероятността една частица да попадне в списъка с агрегиращи е 1-pA2k, a към тази за свърващи се към кинк 
е pA2k.

След това към първия се прилага агрегационно правило, а към втория - кристализационно (към кинк). Резултантните списъци след прилагането на правилата се комбинират в един и той се размесва случайно
