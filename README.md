# FER-2013 სისტემური კვლევა

## პროექტის აღწერა

ამ პროექტში FER-2013 მონაცემთა ბაზაზე სახის ემოციების ამოცნობის კვლევა ჩავატარე. 
კვლევა დავყავი სამ ნაწილად. თითოეულ ნაწილში ვიკვლევდი სხვადასხვა არქიტექტურის გავლენას მოდელის საბოლო შედეგზე

## პროექტის სტრუქტურა

Assignment-4

├── phase1-depth-study/          # პირველი ფაზა: depth კვლევა         
│   ├── 2layer-baseline         
│   ├── 4layer-baseline         
│   ├── 6layer-baseline         
│   └── 8layer-baseline         
├── phase2-skip-connections/     # მეორე ფაზა: skip connections კვლევა         
│   ├── 6layer-standard-cnn         
│   ├── 6layer-skip-connections-cnn         
│   ├── 6layer-hybrid-skip-connections-cnn          
│   └── 6layer-dense-connections-cnn         
├── phase3-regularization/       # ფაზა 3: regularization კვლევა         
│   ├── best-arch-no-reg        
│   ├── best-arch-dropout       
│   ├── best-arch-batchnorm     
│   └── best-arch-both          
└── model-inference/            # მოდელის შემოწმება test set-ზე


## კვლევის ეტაპები

### ფაზა 1: სიღრმის კვლევა
ამ ეტაპზე დავაკვირდი ნეირონული ქსელის სიღრმის გავლენა მოდელის ეფექტურობაზე:
###მოლოდინი: საუკეთესო შედეგი ექნება 4 ფენიან მოდელს

- **2 ფენიანი მოდელი**: იყო ძალიან underfitted
- **4 ფენიანი მოდელი**: შეიმჩნეოდა გაუმჯობესება
- **6 ფენიანი მოდელი**: საუკეთსო შედეგის მქონე აღმოჩნდა
- **8 ფენიანი მოდელი**: იყო overfitted

**შედეგი**: 6 ფენიანი მოდელი იყო ყველაზე ოპტიმალური.



### ფაზა 2: Skip Connections კვლევა
წინა ექპერიმენტიდან გამოვლენილი საუკეთესო არქიტექტურის გაუმჯობესების მიზნით ვცადე სხვადასხვა ვარიანტები
###მოლოდინი: საუკეთესო შედეგი ექნება hybrid skip connection მოდელს

- **სტანდარტული 6 ფენიანი**: ავიღე საბაზისოდ რათა მარტივი დასანახი ყოფილიყო სხვა მოდელების რელატიური შედეგები
- **skip-connections-cnn**: უკვე შეეტყო მოდელს გაუმჯობესება
- **hybrid-skip-connections-cnn**: მიუხედავად იმისა რომ დიდი გაუმჯობესება არ იყო validation accuracy-ში ეს მოდელი
იყო ნაკლებად overfitted ვიდრე წინა
- **skip-connections-cnn**: overfitting-ის მხრივ ამ მოდელმა საუკეთესო შედეგი აჩვენა თუმცა სიზუსტით ჰიბრიდული უკეთესი აღმოჩნდა

**შედეგი**: შემდეგ ეტაპზე გადავედი 6 ფენიანი ჰიბრიდული skip connection არქიტექტურით
უნდა აღინიშნოს რომ ამ ფაზაში ყველა მოდელი overfitted იყო.



### ფაზა 3: რეგულარიზაციის ტექნიკები
წინა ექსპერიმენტების გათვალისწინებით გადავედი სხვადასხვა რეგულარიზაციის ტექნიკების დატესტვაზე იმის იმედით, რომ 
რომელიმე მათგანი overfitting score-ს გააუმჯობესებდა:
###მოლოდინი: საუკეთესო შედეგი ექნება batch normalization-იან მოდელს

- **რეგულარიზაციის გარეშე**: ავიღე საბაზისოდ რათა მარტივი დასანახი ყოფილიყო სხვა მოდელების რელატიური შედეგები
- **Dropout**: შეეტყო მოდელს, რომ რეგულარიზაციის ტექნიკები ჩაერთო საქმეში :დ
- **Batch Normalization**: ამ არქიტექტურამ გაზარდა როგორც მოდელის სიზუსტე validation set-ზე, ისე შეამცირა overfitting score
- **Dropout+Batch Normalization**: რადგან რეგულარიზაციის მეთოდებმა ცალცალკე იმოქმედეს არ იყო გასაკვირი რომ მათი კომბინაცია კარგ შედეგს დადებდა

**შედეგი**: Batch Normalization-იანი მოდელი იყო ყველაზე ზუსტი, თუმცა Dropout+Batch Normalization ჰქონდა უკეთესი შედეგი overfitting-ის შემცირებაში და მისი სიზუსტე არ ჩამოუვარდებოდა პირველს (Batch Normalization-66.07% ხოლო Dropout+Batch Normalization-65.73%), შესაბამისად გადავწყვიტე ეს მოდელი ამეღო საბოლოო prediction-ის გასაკეთებლად.

## მონაცემები

**FER-2013** (Facial Expression Recognition 2013) მონაცემთა ბაზა შეიცავს:
- 35887 48x48 სურათს
- ისინი დაყოფილია 7 ემოციურ კლასად:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral


- ცნობილია რომ Disgust ემოციის ამოცნობა არის რთული ამ მონაცემებში რადგან მისი წილია (1.3%)
ჩემმა საუკეთესო მოდელმა ასეთი შედეგი დადო:
| Emotion | Recall | Samples |
|---------|--------|---------|
| Fear    | 0.310  | 496     |
| Disgust | 0.411  | 56      |
| Angry   | 0.559  | 467     |



### Wandb-ის რეპორტები

#### ფაზა 1: Depth კვლევა
- **სიღრმის შედარებითი ანალიზი**: [Depth Study](https://wandb.ai/skara21-free-university-of-tbilisi-/fer-2013-depth-study/reports/Depth-Study-Report--VmlldzoxMzEwODk3NQ?accessToken=7917x675zw2b6iisbx2h34c7zrtbm9v9g5mc7uqwdcfv7u8jp14xauvl7zy9zaio)

#### ფაზა 2: skip connections კვლევა
- **Skip Connections შედარება**: [Skip Connections Study](https://wandb.ai/skara21-free-university-of-tbilisi-/fer-2013-connections-study/reports/Connection-Study-Report--VmlldzoxMzEwOTE0NA?accessToken=3ilnyff5qfc57lz6u55udqemqxz0paf3t42xgpjvba1eo2i095pqps2yte854q8o)

#### ფაზა 3: რეგულარიზაციის ტექნიკები
- **რეგულარიზაციის შედარება**: [Regularization Study](https://wandb.ai/skara21-free-university-of-tbilisi-/fer-2013-regularization-study/reports/Regularization-Study-Report--VmlldzoxMzEwOTE5Ng?accessToken=qup21w5fth6fygyn7z7kzweidzch15k85df1sst83405no6cvi6eyc8g5cygn4qu)


### დასკვნები:

1. **ოპტიმალური სიღრმე**: 6 layers

2. **გამოტოვებითი კავშირები**: Hybrid Skip Connections CNN

3. **რეგულარიზაცია**: Batch Normalization + Dropout

### მეტრიკები:
- **Accuracy**: **0.6397**
- **F1-Score**: **59%**
- **Precision**: **63%**
- **Recall**: **64%**


#### კლასების მიხედვით დეტალური შედეგები:
| ემოცია | სიზუსტე | Recall | F1-Score | ნიმუშები |
|--------|-----------|--------|----------|----------|
| სიბრაზე | 0.58 | 0.56 | 0.57 | 467 |
| ზიზღი | 0.50 | 0.41 | 0.45 | 56 |
| შიში | 0.46 | 0.31 | 0.37 | 496 |
| ბედნიერება | 0.83 | 0.86 | 0.85 | 895 |
| მწუხარება | 0.51 | 0.57 | 0.54 | 653 |
| გაოცება | 0.76 | 0.82 | 0.79 | 415 |
| ნეიტრალური | 0.57 | 0.61 | 0.59 | 607 |


- **საუკეთესო მაჩვენებელი**: ბედნიერება, რადგან აქვს ყველაზე მეტი ნიმუში
- **ყველაზე სუსტი მაჩვენებელი**: შიში, ხშირად ერევა ბრაზსა და მწუხარებაში
- **კლასების დისბალანსის გავლენა**: ზიზღი ცუდად მუშაობს დიდი ალბათბის იმიტომ რომ, მხოლოდ 56 ნიმუშის აქვს


#### შენიშვნები: 
- WandB-ის ექსპერიმენტები არის private არ ვიცოდი როგორ გამეხადა საჯარო, შემიძლია Team-ში დაგამატოთ, თუმცა რეპორტები შეიცავს ყველა საჭირო გრაფს
- გითჰაბზე ატვირთვისას წავშალე dataset, რადგან ძალიან დიდი იყო და LFS ფაილებთან მუშაობა არ მინდოდა
