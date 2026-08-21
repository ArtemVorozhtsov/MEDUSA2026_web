# MEDUSA 2026 Web Service — техническое описание (для поддержки)

Документ описывает устройство репозитория `medusa_web/`, выбранные технологии,
ключевые проектные решения и «неочевидные» места, без знания которых легко
сломать сервис. Для пользователя (не разработчика) — см. `README.md`.

Источник истины по поведению пайплайна: `MEDUSA2026/research/formula_analysis_examples/entry.ipynb`
и `MEDUSA2026/research/formula_analysis_examples/utils.py`.

---

## 1. Что это

Docker-сервис, позволяющий прогнать полный 7-шаговый пайплайн MEDUSA 2026 по
**одному скану реального FT-ICR масс-спектра** (mzXML) через браузер:

1. Загрузка спектра (файл с сервера/NFS либо upload из браузера)
2. Деизотопирование (группировка пиков в изотопные распределения, заряд)
3. Классификация элементов (Transformer по каждому иону)
4. Определение порога «коленом» (knee) по любому элементу
5. Подсветка ионов выше порога
6. Определение молекулярной формулы (редактируемые ограничения по элементам)
7. Сравнение выбранной формулы с экспериментальным спектром

Реальный тестовый спектр: `MEDUSA2026/spectra/AV-14_MS2_15min_000001.mzXML`
(112 МБ, **7 959 252 точки**, m/z 153.5–3000). Все требования по
производительности (≤2500 точек на ответ, зум <300 мс, полный прогон ~15 с)
проверены на нём.

---

## 2. Структура репозитория

`medusa_web/` — **отдельный новый git-репозиторий**, лежит рядом с основным
репозиторием `MEDUSA2026/` (сibling). Всё, что принадлежит сервису, — только
внутри `medusa_web/`; в `MEDUSA2026/` ничего не меняется (запрет задачи).

```
medusa_web/
├── Dockerfile                # self-contained image (python:3.9-slim)
├── docker-compose.yml        # один сервис, порт ${PORT:-8000}, монтирование спектров
├── requirements.txt          # курируемый подмножество зависимостей (pinned)
├── .dockerignore             # build context = РОДИТЕЛЬСКИЙ каталог (см. §8)
├── Dockerfile.dockerignore   # -> symlink на .dockerignore (BuildKit требует имя <Dockerfile>.dockerignore)
├── README.md                 # документ для пользователя
├── Readme_tech.md            # этот файл
├── .gitignore
├── app/                      # FastAPI-приложение (единственный worker)
│   ├── main.py               # create_app(), lifespan: загрузка ОБОИХ моделей один раз,
│   │                         #   /healthz, монтирование static/ (последним — чтобы /api/* имело приоритет)
│   ├── config.py             # Settings из env (lru_cache + reset_settings_cache() для тестов)
│   ├── state.py              # Session / SessionStore: в-памяти состояние сессий
│   ├── jobs.py               # CpuGate + run_cpu_job (ограничение параллельных CPU-шагов, 409)
│   ├── errors.py             # иерархия ApiError + JSON-обработчики (включая catch-all)
│   ├── engine/
│   │   ├── spectrum_io.py    # Experiment() загрузка (1-й скан), листинг *.mzXML (глубина ≤3)
│   │   ├── downsample.py     # downsample_window (peak-preserving) и uniform_window (для compare)
│   │   └── pipeline.py       # шаги 1–7 как чистые функции (store, session_id, params) -> dict
│   └── api/
│       ├── files.py          # GET /api/files, POST /api/upload (лимит MAX_UPLOAD_MB, 413)
│       ├── sessions.py       # POST/GET/DELETE /api/sessions, GET /api/elements
│       ├── analysis.py       # deisotope / elements / knee / threshold / formulas / presets
│       └── figures.py        # GET .../spectrum (окно+слои), GET .../compare (plotly-fig)
├── static/                   # фронтенд: один HTML-файл, vanilla JS, без build-шага
│   ├── index.html            # вёрстка: левая колонка (шаги 1–7 + Help), центр (вьювер),
│   │                         #   правая (таблица ионов, knee, формулы, лог)
│   ├── app.js                # всё состояние и логика UI (глобальный объект S)
│   ├── styles.css            # стили: layout grid 360px/1fr/440px
│   └── plotly-2.35.2.min.js  # plotly.js 2.35.2, ОФИЦИАЛЬНЫЙ бандл (byte-for-byte =
│                             #   cdn.plot.ly/plotly-2.35.2.min.js, сверено md5),
│                             #   vendored локально: без CDN (LAN/офлайн)
└── tests/
    ├── conftest.py           # пути, фикстуры api_client/small_spectrum, write_mzxml()
    ├── data/
    │   ├── test.h5           # 8 срезов data/formula_determination_test/test.h5 (по 100k точек)
    │   └── test.pkl          # gzip-pickle GT-формул (sample 0 = C30H49N9O9)
    ├── test_downsample.py    # свойства дексимации (оба алгоритма)
    ├── test_knee.py          # knee на синтетических векторах известных форм
    ├── test_pipeline_smoke.py# полный прогон на малом спектре + guard empty-peak_indices
    │                         #   + compare-фигура + сортировка knee
    └── test_api.py           # TestClient: полный round-trip всех эндпоинтов, upload,
                              #   ошибки, лимит сессий, JSON-ошибка на «шумовом» спектре
```

### Что берётся из `MEDUSA2026/` (изменений там нет)

| Что | Откуда | Как используется |
|---|---|---|
| `mass_automation` (пакет) | `MEDUSA2026/mass_automation` | ядро: `Experiment/Spectrum`, `MlDeisotoper`, `TransformerModel`, `formula_generator_parallel`, `check_presence`, `Formula`, `ELEMENT_DICT`; **для этого проекта patched**: `check_formula.py` (якорные окна пиков) + `plot.py` (см. §12 п.16) |
| CGB-модель деизотопирования | `MEDUSA2026/data/models/charge1_big_optuna150.pkl` (28.5 МБ) | запекается в image, `ENV CGB_MODEL` |
| Чекпоинт Transformer | `MEDUSA2026/nn_models/transfomer_classifier.ckpt` (57.6 МБ) | запекается в image, `ENV TRANSFORMER_CKPT` |
| Тестовые данные | `MEDUSA2026/data/formula_determination_test/` | **срезы** (первые 8 сэмплов) скопированы в `tests/data/` — image не зависит от оригинальных 107 МБ |
| Реальные спектры | `MEDUSA2026/spectra/` (и любой NFS) | монтируются read-only в `/data/spectra` через `SPECTRA_DIR` |

`PYTHONPATH=/app:/app/medusa_web`: в image `mass_automation` лежит в `/app`,
а `app` (веб-код) — в `/app/medusa_web`. В dev-venv те же пакеты резолвятся
через `conftest.py` (масса_automation из sibling `MEDUSA2026/`).

---

## 3. Стек и версии

| Компонент | Версия | Почему |
|---|---|---|
| Python | 3.9 (image `python:3.9-slim`) | venv репо — 3.9.24; торч-чекпоинт и модели собраны под него |
| FastAPI / uvicorn | 0.115.14 / 0.34.3 | синхронные эндпоинты + `asyncio.to_thread` для CPU; **worker ровно 1** (состояние в памяти процесса) |
| torch | 1.12.0 CPU (PyPI) | pin репо; GPU — вне v1 |
| pyopenms | 3.2.0 | читает mzXML (`Experiment`) |
| catboost | 1.2.8 | модель деизотопирования (CGB) |
| plotly (py) | 5.24.1 | pin репо; лимиты версии — в §12.12 |
| numpy/scipy/sklearn/h5py | 1.26.4/1.13.1/1.3.1/3.12.1 | pins репо |
| IsoSpecPy | 2.2.3 | sdist: компилируется одним C++ файлом → в image нужен `build-essential` |
| plotly.js | 2.35.2 (официальный бандл, vendored в `static/`) | нет CDN на университетской сети; совместим с figure-JSON от plotly.py 5.24.1 |
| Фронтенд | vanilla JS + hand-written CSS | решение №10 задачи: без framework и без build-шага |

`requirements.txt` — **курируемое подмножество** зависимостей репо (без jupyter,
xgboost, lightgbm, lightautoml, shap, cupy и т.п. — см. задачу §4). Все версии
pinned к venv репо, чтобы поведение ядра совпадало с нотебуком.

---

## 4. Архитектура

### 4.1 Поток запроса

```
браузер (static/app.js)
   │  fetch /api/...
   ▼
FastAPI router (app/api/*.py)
   │  ApiError -> JSON {"detail": ...} (включая catch-all Exception -> 500 JSON)
   ▼
run_cpu_job()  (app/jobs.py)
   │  CpuGate.try_acquire()  ── busy ──► 409 {"detail": "job slot busy, retry"}
   │  asyncio.to_thread(fn, ...)   (worker-поток, пул по умолчанию)
   ▼
pipeline.<step>(store, session_id, params, models...)
   │  cache по params_hash -> "reused": true (повтор без пересчёта)
   ▼
session (app/state.py)  — массы/интенсивности, ion_id, ion_probs, кэши шагов
```

### 4.2 Модули и их ответственность

- **`app/config.py`** — `Settings` (dataclass) из env: `SPECTRA_DIR`,
  `UPLOAD_DIR`, `MAX_UPLOAD_MB` (300), `MAX_ACTIVE_SESSIONS` (8),
  `MAX_CONCURRENT_CPU_JOBS` (2), `DELETE_UPLOADS_ON_SESSION_RESET` (false),
  `PORT` (8000), `CGB_MODEL`, `TRANSFORMER_CKPT`.
  Флаг: при `DELETE_UPLOADS_ON_SESSION_RESET=true` удаление сессии
  (`DELETE /api/sessions/{id}`) дополнительно удаляет её uploaded-файл
  (только `source=upload`; файлы из `SPECTRA_DIR` не трогаются).
  `get_settings()` — `lru_cache`; тесты вызывают `reset_settings_cache()`.
  Порядок поиска моделей: env → `/app/models` (image) → dev-checkout.
- **`app/state.py`** — `Session` (одна сессия = один mzXML, первый скан):
  float64 `masses`/`ints`, объект `Spectrum`, кэши шагов
  (`deisotope_cache`, `elements_cache`, `knee_cache`, `formulas_cache`;
  кэши шагов 3–6 метятся `for_deiso` — хешем текущей деизотопии, и
  инвалидируются при её перезапуске с другими параметрами),
  `ion_probs` (n_ions × 119, `ion_probs_for_deiso` — хеш, для которой
  посчитана), `point_probs` (на точку, для подсветки),
  лог-буфер (deque 200), `threading.Lock`. `SessionStore` — dict + лимит
  активных сессий (409) + LRU-метки; при delete обнуляет ссылки на большие
  массивы (свободит память). **Всё состояние в памяти процесса** → рестарт
  контейнера = потеря сессий (документировано, вне v1 — персистентность).
- **`app/jobs.py`** — `CpuGate`: атомарный счётчик (не `asyncio.Semaphore` —
  на Python 3.9 `wait_for(sem.acquire(), 0)` всегда отменяет acquire, см. §12.7).
  `run_cpu_job()` — try_acquire → `asyncio.to_thread` → release в finally.
- **`app/errors.py`** — `ApiError`-иерархия: `BadRequest` (400), `NotFound`
  (404), `SessionLimit`/`JobBusy` (409), `PipelineError` (4xx/5xx с
  человекочитаемым сообщением), `ModelUnavailable` (503). Обработчики:
  `ApiError → {"detail": ...}`; **catch-all `Exception → 500 JSON`** —
  контракт «все ошибки JSON» держится даже для внутренних сбоев ядра.
- **`app/engine/pipeline.py`** — сердце. Чистые функции
  `(store, session_id, params) -> dict` с кэшированием по `params_hash`
  (повтор с теми же параметрами мгновенный, `reused: true`). Каждый шаг
  обёрнут в guards (см. §6). Модели сюда приходят из `app.state`
  (загружаются в lifespan `main.py` — **импорт engine-модулей без side
  effects**, это проверяется тестами).
- **`app/engine/spectrum_io.py`** — `Experiment(path, verbose=False)`, берётся
  **первый скан** (мультискан-файлы: `n_scans` показывается в UI, остальные
  сканы игнорируются — решение №3 задачи). Листинг файлов: `*.mzXML`,
  глубина ≤ 3, относительные пути + размеры.
- **`app/engine/downsample.py`** — два алгоритма (см. §7).

### 4.3 Сессии и кэши

- Сессия создаётся при загрузке спектра (`POST /api/sessions`), id —
  12-значный hex. Ограничение — `MAX_ACTIVE_SESSIONS` (по умолчанию 8),
  при переполнении 409 с подсказкой (кнопка New session / DELETE).
- Каждый тяжёлый шаг кэшируется **по хэшу параметров**:
  deisotope(параметры), elements(элемент), knee(элемент),
  formulas(ion_id + элементные лимиты + ppm + workers). Изменение любого
  параметра → пересчёт; неизменный → мгновенный повтор.
- Связность кэшей: `elements`/`knee` помечены `for_deiso`-хэшем — после
  повторного деизотопирования с новыми параметрами они пересчитываются
  автоматически; `ion_probs` (полная классификация по всем 119 элементам)
  считается один раз на деизотопирование, смена «активного элемента» — просто
  выборка столбца.

---

## 5. Пайплайн: шаги 1–7 (соответствие нотебуку)

| Шаг | Функция | Что делает | Ключевые детали |
|---|---|---|---|
| 1 | `load_spectrum` | `Experiment()` → скан 0 → `masses`/`ints` float64 + `Spectrum` | ~1.5 с на 8M точек |
| 2 | `deisotope` | `MlDeisotoper().load(CGB_MODEL)(spectrum, algorithm, z_max, min_distance, threshold, delta, n1, n2)` | дефолты = нотебук: `adaptive, z_max=3, 0.01, 0.15, 0.007, n1=2, n2=6`; на выходе `ion_id` на точку (-1 = шум) + `ion_info` (m/z **base peak = argmax интенсивности**, заряд = среднее, n_peaks) |
| 3 | `classify_elements` | по каждому иону: `RealIsotopicDistribution.get_representation(f=np.mean, mode="middle", length=101)` → `center_representation(repr, mass, charge_mean)` → нормировка на глобальный max → forward Transformer (батч (n_ions, seq, 100)) → sigmoid | **input_dim=100** (get_representation length=101 → vectorize отбрасывает центр → n_bins−1=100), **output_dim=119** = `N_ELEMENTS`; столбец c ↔ атомный номер c+1 (`ELEMENT_DICT` — 118 имён, у модели класс на один больше); каждый ион в try/except — см. §6 |
| 4 | `knee` | `find_knee_threshold_numpy` (локальная копия утилиты нотебука — без import-path coupling в `research/`): точка, дальше всех от хорды (0,max)→(N,min) **в отсортированном по убыванию** массиве | API возвращает `probs` **отсортированными desc**; `knee_idx` — позиция в отсортированном массиве (чарт в UI рисует ровно это) |
| 5 | `apply_threshold` | `source=auto` (knee) или `manual` (manual_value∈(0,1)); строит `point_probs` (на точку через ion_id) | результат — слой подсветки в вьювере («красные точки») |
| 6 | `formulas` | `target_mass = (mz_base + ELECTRON_MASS)·charge`; `formula_generator_parallel(ELEMENTS, low/high_limits, target_mass, threshold_ppm, num_workers, max_chunk_size, mode="max")` → каждый кандидат: `Formula(...) + check_presence` в `joblib.Parallel(n_jobs=num_workers, loky)` (num_workers ограничен `MAX_FORMULA_WORKERS = 16`, как и генерация) → ранжирование **cosine desc, потом delta asc** | `cosine = 1 − cosine_distance` (сходство); ошибки генератора/валидации → 4xx/5xx с сообщением, никогда не «тихо пусто» |
| 7 | `compare` | окно `[theo_masses[0]−1, theo_masses[−1]+1]`; `Formula.isodistribution()` + `del_isotopologues` + `check_presence` → Plotly-фигура 2×1 | окно — `uniform_window` (формосохраняющий), не peak-preserving; окно небольшое и фиксировано сервером → уходит **точным** (недексимированным), пока ≤ `COMPARE_HARD_MAX_PTS` (40k), иначе uniform step≥1 **+ merge сырых окрестностей ±0.01 Da вокруг matched-пиков** — линия гарантированно проходит через все matched-пики (см. §7); стиль calculated = вертикали 0→rel.intensity + m/z-подписи (как в `plot_compare`); метрики (Δ, cos. dist., matched %) — в рамке справа сверху |

## 6. Guards против известных (латентных) багов ядра

Задача требует **не чинить ядро**, а оборачивать каждый пер-ионный вызов:

1. **empty `peak_indices`** (`RealIsotopicDistribution.get_representation` →
   `ValueError: max() arg is an empty sequence`): в `_classify_all_ions`
   каждый ион в try/except → ион помечается пропущенным, в лог пишется
   warning, сессия продолжает. Тест: `test_empty_peak_indices_guard`
   (эмуляция бага через monkeypatch).
2. **`find_peaks` падает на вырожденных спектрах**
   (`IndexError: arrays used as indices must be of integer type` —
   float-индексы пиков, воспроизводится на чистом шуме): вызов `deisotoper(...)`
   в `deisotope()` обёрнут в try/except → `400 {"detail": "deisotoping failed: ..."}`.
   На реальных FT-ICR спектрах не воспроизводится. Тест:
   `test_unprocessable_spectrum_returns_json_error`.
3. **catch-all** в `errors.py`: любой непредвиденный исключение → 500 JSON
   (раньше был plain-text «Internal Server Error» — нарушение контракта API).

Единственное **разрешённое** изменение ядра (задача §10, опционал) — guard в
`mass_automation/formula/__init__.py` — **не делалось**: engine-guardов
достаточно.

---

## 7. Работа с большими спектрами (критическое требование)

Для интерактивного вьювера (`/spectrum`) в браузер **никогда** уходит ≤
`max_pts` точек (дефолт 2500, hard cap 5000 — `Query(le=5000)`; ограничение
задачи §7 привязано именно к pan/zoom-циклу). Статичная compare-фигура —
исключение: её окно фиксировано сервером и мало, поэтому там свой лимит
`COMPARE_HARD_MAX_PTS = 40000` (см. ниже).

### `downsample_window` (основной вьювер) — peak-preserving

- окно ≤ max_pts → точные точки, `decimated: false`;
- иначе: (a) топ-K по интенсивности, K = max_pts//2; (b) остаток окна делится
  на (max_pts−K)//2 подряд идущих бинов, из каждого берутся max **и** min;
  (c) merge + сортировка.
- Гарантии (тестируются): m/z строго возрастают; ≤ max_pts; **глобальный max
  окна всегда на месте**; концы окна представлены с точностью до ширины бина.
- Вспомогательные массивы (`ion_id`, `prob`) режутся и дексимируются **параллельно**
  (остаются выровненными с `masses`).
- `_first_extremum`: sentinel = `np.iinfo(int64).max` (не −1! −1 ломает
  `np.minimum.at`).

### `uniform_window` (только compare) — shape-preserving

Peak-preserving дексимация **искажает относительные интенсивности** (смещает
в пользу пиков) — для наложения расчётного изотопного паттерна это недопустимо.
Поэтому compare-окно строится равномерной выборкой «каждый N-й» + обязательное
включение argmax (с контролем лимита ≤ max_pts: при переполнении выбрасывается
ближайшая к argmax выброчная точка). Результат визуально совпадает с реальным
спектром (проверено тестом: максимум compare-окна == максимум настоящего окна).

**Разрешение compare-окна** (`pipeline.compare`): окно = кластер ±1 Da
(на реальном спектре 7,96M точек это ~10–60k сырых точек). Политика:
- `max_pts` не передан (дефолт) → `min(n_in_window, COMPARE_HARD_MAX_PTS)`:
  окно ≤ 40k уходит **точным** (`decimated=false`, step=1) — линия проходит
  ровно через вершины всех matched-пиков (они находятся в полном разрешении
  через `check_presence`); > 40k — uniform со step ≥ 1;
- `max_pts` передан явно (legacy) — respected как cap (clamp до 40k).
**Merge окрестностей matched-пиков** (гибрид): плотные низкомассовые
регионы могут содержать >100k сырых точек в несколько Da (замер: окно
[165, 168] = 139 411 точек). Увеличивать cap до 150k нецелесообразно
(payload ~4.5 МБ, SVG-рендер тяжёлый), поэтому при дексимации compare-окна в
выборку дополнительно включаются сырые окрестности ±`MATCHED_NEIGHBOR_HALF_WIDTH`
(0.01 Da) вокруг каждой matched-массы (индексы через `searchsorted` по срезам
окна, параметр `extra` у `uniform_window`). Тогда линия гарантированно
проходит через вершины **всех** matched-пиков (ищите их — кружки из
`check_presence` находятся в полном разрешении), а число точек остаётся
≤ 40k + ~5k. Точное окно (< 40k) merge не нужно — там все точки на месте.

**Вставка самих точек-кружков** (`_merge_matched_points`): `check_presence`
при **не**найденном пике (слабые изотопологи ниже порога `find_peaks`)
возвращает **теоретическую** массу как fallback — это не сырая точка
спектра, и в точном окне она могла отсутствовать на линии (кружок в
подвесе). Поэтому координаты всех matched-точек (масса+интенсивность из
`real_coords`) вставляются в экспериментальный трек (стабильная сортировка
по m/z, точные дубликаты отбрасываются — выигрывает сырая точка). Кружок
лежит на линии **по построению**, в любом режиме окна.

Замер: окно [972, 980] (10 818 точек) — точная отрисовка стоит ~2 мс CPU и
~0,35 МБ JSON; доминирующая доля времени compare — `check_presence`, от
`max_pts` не зависит. 40k выбрано с запасом: SVG-рендер Plotly комфортен до
~40–50k точек в статичной фигуре (сверх — `scattergl`).

### Вьювер (фронтенд)

- первичная загрузка: один запрос на весь диапазон (дексимированный обзор);
- `plotly_relayout` (drag/scroll zoom) → **debounce 150 мс** → fetch нового окна;
- слои: `raw` (всегда, синяя линия), `ions` (точки цвета иона **на каждой точке
  иона в окне** — один цвет на изотопное распределение; hover: номер иона, m/z,
  I, base peak), `probs` (красная подсветка выше порога);
- легенда вьювера **не** выводит список ионов (только запись highlight-слоя);
- клик по точке иона или пику (через параллельный `ion_id`) → выбор иона
  + авто-зум в его окно;
- `window_range` в ответе — для статуса («m/z x–y» в строке состояния).

---

## 8. Docker

### 8.1 Build context — ВАЖНО

Context — **родительский** каталог (`medusa_web/` и `MEDUSA2026/`), т.к.
Dockerfile копирует `mass_automation` и модели из `MEDUSA2026/`:

```yaml
build:
  context: ..
  dockerfile: medusa_web/Dockerfile
```

Поэтому `.dockerignore` живёт в `medusa_web/`, но BuildKit ищет его как
`<Dockerfile>.dockerignore` в корне context → сделан **symlink**
`medusa_web/Dockerfile.dockerignore → .dockerignore`. Если кто-то перенесёт
Dockerfile — symlink и селекторы игнора (`MEDUSA2026/mass_automation` и т.д.)
перестанут работать. Без него context утягивает 4 ГБ `.venv` и 107 МБ спектров.

### 8.2 Слои image (порядок имеет значение для кэширования)

1. `apt`: `build-essential` (g++ для sdist IsoSpecPy) + **`libglib2.0-0[t64]`**
   (pyopenms при импорте требует `libglib-2.0.so.0`/`libgthread-2.0.so.0`;
   на trixie пакет называется `libglib2.0-0t64`, на bookworm — `libglib2.0-0`,
   в Dockerfile — fallback через `||`);
2. `pip install -r requirements.txt` (кэшируемый слой — меняется редко);
3. **правка ELF**: Python-скрипт снимает `PF_X` у program header
   `PT_GNU_STACK` (0x6474E551) во всех `.so` под site-packages и /app —
   **обязателен** после любого изменения требований (см. §12.1);
4. `COPY` веб-кода, `mass_automation`, моделей;
5. `ENV PYTHONPATH=/app:/app/medusa_web`, пути моделей, лимиты;
6. не-root пользователь `medusa`, `CMD uvicorn ... --workers 1`.

### 8.3 compose

- `container_name: medusa2026-web` (фикс — иначе `up` после `run` создавал
  дубликаты с суффиксами);
- порт `${PORT:-8000}:8000`;
- `${SPECTRA_DIR:-./spectra}:/data/spectra:ro` — NFS-папка со спектрами
  (read-only, сервис не пишет туда);
- named volume `medusa_uploads:/data/uploads` (загруженные файлы переживают
  рестарт контейнера);
- `restart: unless-stopped`;
- `healthcheck` — GET /healthz (python urllib, start_period 60 с — модели
  грузятся ~5–10 с, но на медленных хостах дольше).

Запуск с NFS: `SPECTRA_DIR=/mnt/nfs/medusa_spectra docker compose up --build`
(или в `.env` рядом с compose-файлом).

---

## 9. API (контракт)

Все ошибки — JSON `{"detail": "человекочитаемое сообщение"}`:
400 (валидация/состояние шагов), 404 (сессия/файл), 409 (лимит сессий,
занят слот CPU — «job slot busy, retry»), 413 (upload больше лимита),
500 (внутреннее, с пометкой `internal error: ...`), 503 (модель не загрузилась).

| Метод и путь | Назначение | Ключевое в ответе |
|---|---|---|
| `GET /api/files` | список `*.mzXML` под `SPECTRA_DIR` (глубина ≤3) | `files: [{path, name, size}]` |
| `POST /api/upload` | multipart, ≤ `MAX_UPLOAD_MB` | `file_id` (10 hex + исходное имя) |
| `POST /api/sessions` | `{"source": "folder"|"upload", "path"|"file_id"}`; защита от path traversal (realpath-префикс) | `session_id, n_points, mz_range, n_scans, load_time_s, logs` |
| `GET /api/sessions/{id}` | метаданные + `completed_steps` | |
| `DELETE /api/sessions/{id}` | удалить (освобождает память) | |
| `GET /api/sessions/{id}/spectrum?x0&x1&max_pts&layers` | окно ≤ max_pts; `layers=raw,ions,probs` | `masses, ints, ion_id?, prob?, decimated, n_in_window, full_range, window_range` |
| `POST /api/sessions/{id}/deisotope` | DeisotopeParams (тело может быть `{}` — дефолты нотебука) | `n_ions, ions[], elapsed_s, reused` |
| `POST /api/sessions/{id}/elements` | `{"element": "Ir"}` | `rows: [{ion_id, mz, charge, n_peaks, prob}]` (отсортировано по prob desc), `skipped[]` |
| `GET /api/sessions/{id}/knee?element=Ir` | | `probs` (sorted desc!), `threshold, knee_idx` |
| `POST /api/sessions/{id}/threshold` | `{"element", "source": auto|manual, "manual_value"}` | `threshold, n_points_above, n_ions_above, mz_range` |
| `POST /api/sessions/{id}/formulas` | `ion_id, elements {El: [lo, hi]}, mass_threshold_ppm, num_workers (1…16, иначе 422), max_chunk_size` | `ranked: [{rank, formula, mass, delta_ppm, cosine}], n_candidates, n_valid, n_failed, skipped_pct, elapsed_s, reused` |
| `GET /api/sessions/{id}/compare?ion_id&formula[&max_pts]` | Plotly figure JSON; `max_pts` опционален (дефолт: точное окно ≤ 40k, иначе uniform step≥1) | `data[]` (exp, matched, vlines, подписи m/z), `layout.annotations` (2 заголовка сабплотов + блок метрик) |
| `GET /api/formula_presets` | `ir_system`, `pubchem10`, `empty` | |
| `GET /api/elements` | 118 символов `ELEMENT_DICT` (правильный case: Ir, Cl…) | |
| `GET /healthz` | liveness + `models: {cgb, transformer}` + счётчики сессий | |

Документация: `/api/docs` (Swagger) и `/api/openapi.json`.

**Compare-фигура** (строится на сервере, рендерится браузером):
row 1 — экспериментальное окно (черная линия; `uniform_window`: точное
окно ≤ `COMPARE_HARD_MAX_PTS`, иначе uniform step≥1 + merge сырых
окрестностей ±0.01 Da вокруг matched-пиков) + оранжевые «Matched peaks»
из `check_presence` (полное разрешение — линия проходит через все их
вершины); row 2 — расчётный изотопный паттерн
(черные вертикали 0→rel.intensity по одному на изотоп + m/z-подписи);
блок метрик (Δ, Cos. dist., Matched %) — рамка в правом верхнем углу
(paper-координаты, чтобы не перекрывать данные). Заголовок —
`<formula> (z = N)`. В легенде только «Experimental» и «Matched peaks».

---

## 10. Фронтенд

Одна страница, **English** (решение №6 задачи), desktop-only.
`grid: 360px | 1fr | 440px`.

- **Состояние** — глобальный объект `S` (session, fileInfo, ions, ionProbs,
  selectedIon, element, threshold, knee, ranked, window, elementSymbols) +
  `currentRange` (последний запрошенный x-диапазон).
- **Левая колонка** — Help (свёрнутый `<details>` с описанием шагов) + шаги
  1–7: у каждого — спиннер (`.spinner[data-spin]`), summary-блок и кнопки Run.
- **Центр** — статус-строка, `#spectrum-plot`, статус compare, `#compare-plot`
  (hidden до запуска шага 7; `min-height: 560px`, центральная колонка
  скроллится).
- **Правая** — таблица ионов (сортировка по колонкам, клик = выбор),
  knee-чарт (log-y, sorted desc), таблица ранжированных формул + CSV,
  **лог** (последние 200 записей сессии; очищается при New session).
- **Слой ионов** — чекбоксы `#ions-toggle` (в шаге 2) и `#layer-ions`
  (в viewer-bar) синхронизированы; запрос окна всегда включает `ions`
  после деизотопирования; если слой включён, а в текущем окне нет `ion_id` —
  авто-перезапрос.
- **Зум** — `plotly_relayout` → debounce 150 мс → `refreshWindow`.
  `onPlotClick` — выбор иона (маркер: `trace["medusa-ion"]`; сырой спектр:
  `window.ion_id[pointNumber]`).
- **Toast** — единый `#toast` для ошибок/уведомлений.
- Plotly.js vendored (`static/plotly-2.35.2.min.js`, ~4.5 МБ) — работает без сети.

Правила для правок UI:
- не использовать синтаксис > ES2017 без проверки (базовый современный
  браузер; `node --check static/app.js` — минимум контроля синтаксиса);
- ответы API — только JSON (см. §9), ошибки читать из `data.detail`;
- не выносить данные сессии в localStorage/cookie (всё в памяти сервера).

---

## 11. Тесты

**41 тест**, зелёные и в dev-venv, и внутри image (≈80 с / ≈50 с).

- `conftest.py`:
  - находит корень `mass_automation` (env `MASS_AUTOMATION_PATH` → sibling
    `MEDUSA2026/` → `/app` в image) и подставляет модели из env
    (`CGB_MODEL`/`TRANSFORMER_CKPT`; в image — пути из ENV, в venv — файлы
    репо); `SPECTRA_DIR`/`UPLOAD_DIR` по умолчанию — `tests/data/`;
  - `api_client` — TestClient с **изолированными** tmp-каталогами и
    `MAX_ACTIVE_SESSIONS=4`; модели грузятся один раз (lifespan);
  - `write_mzxml(path, masses, ints)` — ручной writer mzXML (sashimi 3.2),
    т.к. pyteomics не умеет писать (см. §12.5).
- `test_downsample.py` — свойства обоих алгоритмов (§7), включая
  shape-preservation `uniform_window` (нет «выпадения» кусков, argmax на месте,
  точные значения исходных точек).
- `test_knee.py` — knee на известных формах (elbow, экспоненциальный спад,
  константа, короткие векторы).
- `test_pipeline_smoke.py` — полный прогон на sample 0 (100k точек):
  деизотоп → elements → knee → formulas; GT-формула (`C30H49N9O9`) обязана быть
  в ranked; кэш; **guard empty-peak_indices**; compare-фигура
  (лимит точек, сохранение максимума окна, метрики с Δ); сортировка knee.
- `test_api.py` — round-trip всех эндпоинтов через HTTP (upload и folder),
  ошибки (404/400/409), **JSON-ошибка на шумовом спектре** (дегенеративный
  вход ядра), лимит сессий.

Запуск:

```bash
# dev venv (из родительского каталога medusa_web/)
pytest medusa_web/tests -q
# внутри image
cd medusa_web && docker compose run --rm web pytest medusa_web/tests -q
```

Тесты грузят реальные модели (нужны `CGB_MODEL`/`TRANSFORMER_CKPT`), поэтому
без venv/image медленны — это осознанно (smoke-режим против unit-моков).

---

## 12. Неочевидные места и «почему так»

1. **glibc 2.41 и execstack (критично для rebuild).** `python:3.9-slim`
   теперь строится на Debian 13 (trixie), glibc 2.41 отказывается работать с
   `.so`, у которых `PT_GNU_STACK` имеет флаг X:
   `ImportError: libtorch_cpu.so: cannot enable executable stack as shared
   object requires`. Единственный такой файл — `torch/lib/libtorch_cpu.so`
   (wheel 1.12.0). В Dockerfile после pip снимается `PF_X` у `PT_GNU_STACK`
   (константа **0x6474E551**, не 0x64!). Если меняется requirements — шаг
   обязан остаться после `pip install` (иначе слой кэширования «съест» fix).
   Хост (Ubuntu 24.04, glibc 2.39) этот флаг терпит — поэтому в venv всё
   работало, а в image нет.
2. **pyopenms в image**: wheel бандлит зависимости в `pyopenms.libs/`
   (conda-style имена `libQt5Core-023a4552.so...`) с RPATH
   `$ORIGIN/../lib/:$ORIGIN/../pyopenms.libs`. `ldd` по отдельным файлам из
   `pyopenms.libs/` показывает «not found» — это **артефакт** (ldd запускает
   без «родительского» RPATH); реальная цепочка импорта резолвится. Реально
   не хватает только GLib (пакет из §8.2).
3. **Нельзя менять `MEDUSA2026/`** (задача §10), кроме опционального guard в
   `formula/__init__.py` (не делался). Все обёртки — в `app/engine/`.
4. **Один worker uvicorn** — жёсткое требование: сессии в памяти процесса.
   `--workers 2` «сработает», но разорвёт состояние (сессии пропадают
   по запросу). Массштабирование — вне v1.
5. **mzXML writer руками** (`tests/conftest.py::write_mzxml`): pyteomics 4.5.3
   не пишет mzXML. Формат критичен для pyopenms 3.2.0:
   - namespace sashimi 3.2;
   - у `<scan>` **обязателен** атрибут `num="1"` (без него pyopenms segfault);
   - `compressedLen` = длина base64; `byteOrder="network"` (big-endian);
   - interleaved m/z-intensity (f8);
   - **не включать** `m/zAverage` (парсер pyopenms его не переносит).
6. **Контракт «все ошибки JSON»** держится тремя уровнями: `ApiError`
   (контролируемые), guards в pipeline (баги ядра), catch-all в `errors.py`.
   Не «лечить» симптом, возвращая 200 с пустыми данными (задача: «never a
   silent empty list»).
7. **Python 3.9**: в сигнатурах FastAPI/Pydantic — только `Optional` (нет
   `X | None`); `CpuGate` вместо `asyncio.Semaphore` — `wait_for(sem.acquire(), 0)`
   на 3.9 всегда отменяет acquire (gate всегда «free»).
8. **`params_hash`** — кэши шагов хеширует **все** параметры dataclass; если
   добавить поле в params — кэши автоматически изменят поведение. Не хэшировать
   вручную отдельные поля. Кроме того, результаты шагов 3–6 привязаны к
   текущей деизотопии (`for_deiso` = `last_deisotope_hash`): повторный запуск
   шага 2 с другими параметрами инвалидирует кэши `elements_cache`/`formulas_cache`
   и `knee_cache`, а `apply_threshold`/`knee` на устаревших `ion_probs`
   отвечают 400 «re-run step 3».
9. **`ion_probs` — 119 столбцов, а не 118**: `ELEMENT_DICT` даёт 118 имён,
   чекпоинт обучен на `output_dim=119`. Столбец `atomic-1`. Не «исправлять»
   размерность под словарь.
10. **Колонки элементов в UI**: `GET /api/elements` отдаёт символы с
    правильным case (Ir, Cl). Раньше CSS `text-transform: uppercase` ломал
    вид — если добавляете новые поля с символами элементов, не применяйте
    uppercase (иначе «IR» в запросы уйдёт не «Ir» — валидация по
    `element_atomic_number` отклонит).
11. **joblib `n_jobs=-1`** в `formulas()` (локо) — валидация кандидатов
    пулит объект `Spectrum` (сотни МБ) на воркеров; на большой сессии это
    заметная память/время. Осознанно по спецификации (нотебук так же).
12. **Лимиты plotly.py 5.24.1** (не поднимать версию — pin репо; figure-JSON
    генерируется на сервере, рендерит браузерный plotly.js 2.35.2):
    - в plotly.PY 5.24.1 нет `go.Vline` (даже если JS-бандл умеет его
      рендерить) и нет `mode="vlines"` → вертикали рисуются трейсами из двух
      точек `[m, m] × [0, rel]`;
    - `update_layout(annotations=[...])` делает **поэлементное слияние**
      (дублирует/затирает), а не замену → править `fig.layout.annotations`
      прямым присваиванием (в `compare()` — append к заголовкам сабплотов);
    - у `go.Scatter` нет `textangle` (только у Bar).
13. **Синхронные «длинные» эндпоинты** — осознанно (решение №7 задачи):
    спиннер + 409 при насыщении пула, без job-очередей (вне v1).
14. **`/healthz`** должен отвечать даже если модель не загрузилась
    (покажет `models: {cgb: false, ...}`); эндпоинты, требующие модель,
    тогда вернут 503. Не блокировать старт контейнера падением моделей.
15. **plotly.js в `static/`** — официальный `plotly-2.35.2.min.js`
    (byte-for-byte совпадает с cdn.plot.ly, md5 `bed2e6700214f7cc365e0f159f538531`).
    При замене версии менять `<script src>` в `index.html`; версии 3.x/2.3x
    совместимы с нашим figure-JSON, но проверить нужно (API `Plotly.newPlot/
    react` и события `plotly_relayout/plotly_click` стабильны).

16. **`check_presence` — поведение ядра (изменено для этого проекта,
    MEDUSA2026 commit `04c7434`)**: окно поиска каждого изотопического пика
    привязано к **теоретической** массе + бегущий сдвиг (медиана
    matched−theo по найденным пикам), а не к предыдущему matched-пику
    (цепочный дрейф при нелинейной калибровке уводил окно за собственный
    пик, и matched «уходил» на чужой пик соседней молекулы — кейс Pd2-димера,
    m/z 1147/1155). `dist_error` по умолчанию 0.006 (было 0.003). Кружки
    «Matched peaks» на compare — это именно выбранные алгоритмом пики:
    для слабых изотопологов, где `find_peaks` пик не нашёл, ядро возвращает
    **теоретическую** массу (fallback) — поэтому в `compare()` сами точки
    вставляются в линию (`_merge_matched_points`), и кружок лежит на линии
    по построению. Метрики (Δ, косинус, matched %) считаются на этих же
    точках — как в нотебуке.

---

## 13. Производительность (измерено на целевой машине, 64 ядра / 188 ГБ)

| Операция | Время |
|---|---|
| `docker compose build` (с нуля, pip+IsoSpecPy) | ~4–5 мин |
| Старт контейнера + загрузка моделей | ~5–10 с |
| Загрузка 8M-спектра (112 МБ) | ~1.5 с |
| Деизотопирование всего спектра (adaptive, z_max=3) | ~0.5–1 с (57 ионов) |
| Элементная классификация (57 ионов, CPU) | ~1–2 с |
| Formulas (ir_system, топ-ион) | ~2–3 с (локо-валидация 30 кандидатов) |
| Полная цепочка 1→7 (E2E, без времени пользователя) | **~12–15 с** (лимит ~2 мин) |
| Запрос окна спектра (любое) | ≤2500 точек; <300 мс (хуже 220 мс на холодном кэше) |
| Тесты: venv / в image | ~80 с / ~50 с (43 теста) |

Память сессии на 8M-спектр: ~2×8M×8 Б (float64 masses/ints) + `ion_id`
(int32) + `ion_probs` — порядок сотен МБ; при 8 сессиях — единицы ГБ.
`SessionStore.delete` обнуляет ссылки для освобождения.

---

## 14. Ограничения v1 (из задачи §14 — не делать)

Аутентификация/мульти-тенант, job-очередь и фоновые воркеры, GPU,
мульти-скан-аналитика (только первый скан), батч-анализ, персистентность
результатов через рестарт, дообучение моделей, i18n (UI — English).

---

## 15. Быстрая шпаргалка

```bash
cd medusa_web
docker compose up --build                       # сборка + запуск (порт 8000)
SPECTRA_DIR=/mnt/nfs/medusa_spectra docker compose up -d   # NFS-папка
docker compose run --rm web pytest medusa_web/tests -q      # тесты в image
curl -s localhost:8000/healthz                  # модели: cgb/transformer
docker logs -f medusa2026-web                   # логи (загрузка моделей, 409, warnings)
docker compose down                             # стоп
```

Полезные env (compose): `PORT`, `SPECTRA_DIR`, `MAX_UPLOAD_MB`,
`MAX_ACTIVE_SESSIONS`, `MAX_CONCURRENT_CPU_JOBS`, `DELETE_UPLOADS_ON_SESSION_RESET`;
внутри image:
`CGB_MODEL`, `TRANSFORMER_CKPT` (override путей моделей).
