

<h1 align="center">DAP: Doppler-aware Point Network for Heterogeneous mmWave Action Recognition</h1>

<p align="center">
  <a href="https://scholar.google.com.hk/citations?user=kU4rtNQAAAAJ&hl=zh-CN">Jiaying Lin<sup>1</sup></a>,
  <a href="https://openreview.net/profile?id=~Shiman_Wu1">Shiman Wu<sup>2</sup></a>,
  <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=jdOJpl0AAAAJ">Jinfu Liu<sup>3</sup></a>,
  <a href="https://scholar.google.com/citations?user=sEeImYgAAAAJ&hl=en">Can Wang<sup>4</sup></a>,
  <a href="https://scholar.google.com.hk/citations?user=woX_4AcAAAAJ&hl=zh-CN">Mengyuan Liu<sup>1</sup></a>
</p>

<p align="center">
  <sup>1</sup>Peking University&nbsp;&nbsp;&nbsp;
  <sup>2</sup>Huazhong University of Science and Technology&nbsp;&nbsp;&nbsp;
  <sup>3</sup>DJI Technology Co., Ltd.&nbsp;&nbsp;&nbsp;
  <sup>4</sup>Kiel University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.09604">📄 Paper</a> |
  <a href="https://github.com/jolin830/DAP-Net">💻 Code</a>
</p>

---

## Dataset

<div align=center>
<img src ="./figs/Intro.png" width="1600"/>
</div>


### Download Links

#### Cross-Subject
- **Dataset (.npz)**: [Google Drive](https://drive.google.com/file/d/1hb0JYzbC3JAx1WfNL4-wKbZFAY0wkHNg/view?usp=sharing)
- **Dataset (.csv)**: [Google Drive]()
- **Normalized Dataset**: [Google Drive](https://drive.google.com/file/d/1ewkzmpGWuVdBjNymiEVS012_Pg7lcEp1/view?usp=sharing)

#### Cross-Set
- **Dataset (.npz)**: [Google Drive](https://drive.google.com/file/d/1tvEprIFL938X6Ax3l9N7mTIx_s0aEl_9/view?usp=sharing)
- **Dataset (.csv)**: [Google Drive]()
- **Normalized Dataset**: [Google Drive]()

#### All
- **Dataset (.npz)**: [Google Drive]()
- **Dataset (.csv)**: [Google Drive]()

---

## Data Preparation

1. Download the original datasets:
   - [RadHAR](https://github.com/nesl/RadHAR)
   - [mRI](https://github.com/SizheAn/mRI)
   - [MM-Fi](https://github.com/ybhbingo/MMFi_dataset)

2. Generate CSV files:
```bash
python dataset/makecsv/RadHAR2csv.py
python dataset/makecsv/mRI2csv.py
python dataset/makecsv/MMFI2csv.py
````

3. Generate NPZ files:

```bash
python dataset/makenpz/makenpz.py
```

4. If normalization is needed:

```bash
python dataset/makenpz/makenpz_normal.py
```

5. Split the dataset according to the rules described in the paper.

---

## Naming Convention

Example:
`D001A001E001P001S0001`

* `D`: dataset identifier
* `A`: action category
* `E`: acquisition environment
* `P`: participant index
* `S`: sample index

If the original dataset does not provide certain fields, the corresponding part can be omitted.

---

## Dataset Directory Structure

```bash
UniMM-HAR/
├── CSub/
│   ├── train/
│   │   ├── D000A003P000S0001.npz
│   │   └── ...
│   └── test/
│       ├── D000A003P010S0084.npz
│       └── ...
├── CSet/
│   ├── train/
│   │   ├── D000A002P000S0001.npz
│   │   └── ...
│   └── test/
│       ├── D001A001E000P000S0001.npz
│       └── ...
└── All/
    ├── MMFI/
    │   └── D001A001E000P000S0001.npz
    │   └── ...    
    ├── RadHAR/
    │   └── D002A000S0001.npz
    │   └── ...
    └── mRI/
        └── D000A000P000S0001.npz
        └── ...
```

---

## Action Classes

| ID | Action |
|---:|---|
| 0 | walk |
| 1 | jump |
| 2 | squat |
| 3 | stretch |
| 4 | jumping jack |
| 5 | box |
| 6 | extend left upper limb |
| 7 | extend right upper limb |
| 8 | extend both upper limbs |
| 9 | left front lunge |
| 10 | right front lunge |
| 11 | left side lunge |
| 12 | right side lunge |
| 13 | extend left limb |
| 14 | extend right limb |
| 15 | expand chest horizontally |
| 16 | expand chest vertically |
| 17 | twist left |
| 18 | twist right |
| 19 | mark time |
| 20 | extend both limbs |
| 21 | raise left hand |
| 22 | raise right hand |
| 23 | wave left hand |
| 24 | wave right hand |
| 25 | pick up object |
| 26 | throw left |
| 27 | throw right |
| 28 | kick left |
| 29 | kick right |
| 30 | extend left body |
| 31 | extend right body |
| 32 | bow |

---

## Code

```bash
coming soon
```

---



