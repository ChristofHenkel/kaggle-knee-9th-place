# Knee Abnormality Detection AI Challenge Overview

---

**Date:** 2026-08-06T15:54:27.876000Z

---

**Courtesy of Dr. Jacob Kazam**

## Overview of the Clinical Problem

The 2026 RSNA Knee Abnormality Detection AI Challenge asks participants to build machine learning models that detect a defined set of clinically important abnormalities on knee MRI examinations. It is the first RSNA AI Challenge to pair every imaging study with the text of its original radiology report, and to do so across a large, multilingual, international dataset. Models may learn from both the images and the report text, reflecting how knee MRI is actually interpreted and communicated in clinical practice.

The knee is one of the most frequently injured and most frequently imaged joints in the body. Knee disorders are a major and growing cause of pain and disability worldwide, affecting athletes, manual workers, and older adults alike. The knee is the most common site of osteoarthritis, which affects nearly a quarter of adults aged 40 and older, an estimated 654 million people worldwide. Acute injuries are also common: knee injuries account for an estimated 15 to 40 percent of all sports-related injuries.

Several structures are commonly involved. Anterior cruciate ligament (ACL) tears are among the most frequent sports injuries, often occurring during sudden pivots or changes of direction and leading to immediate instability. Medial collateral ligament (MCL) injuries typically result from outward (valgus) stress on the knee and frequently accompany other structural damage. Meniscal tears, which may be traumatic or degenerative, disrupt the joint’s shock absorption and mechanics, producing pain, locking, and functional limitation. Cartilage loss, joint effusion, synovitis, bone contusions, fractures, and Baker (popliteal) cysts are additional findings that cause symptoms and influence management.

MRI is central to evaluating the knee because it depicts all of these structures, including ligaments, menisci, articular cartilage, and bone, without ionizing radiation, providing a comprehensive whole-joint assessment that other modalities cannot match. However, MRI interpretation varies across readers and practice settings, and access to subspecialty-trained musculoskeletal (MSK) radiologists is limited in many parts of the world. Reliable, consistent automated detection of knee abnormalities could help extend expert-level interpretation, reduce variability, and support more timely care.

The goal of this competition is to develop machine learning models that detect twelve clinically important knee abnormalities, learning from a large international dataset of knee MRI examinations paired with their original, multilingual radiology reports.

## Anatomical Overview

The knee is a synovial hinge joint formed by three bones: the femur (thigh bone) above, the tibia (shin bone) below, and the patella (kneecap) in front. The ends of these bones are covered by articular cartilage, a smooth layer that allows near-frictionless motion and distributes load. The joint is enclosed by a capsule lined with synovium, which produces lubricating fluid; an abnormal excess of this fluid is termed a joint effusion, and inflammation of the synovium is termed synovitis.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2Fe6095eef091115d6f20f3d5c071dd72e%2Fimage1.jpg?generation=1786032010214772&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2Fc6429a733527b6dd080a84a472a6a34c%2Fimage2.jpg?generation=1786032025815728&alt=media)

*Knee anatomy: anterior view (top) and lateral view (bottom), showing the femur, tibia, patella, cruciate ligaments, and meniscus. Illustrations by BruceBlaus, CC BY 3.0, via Wikimedia Commons.*

For the purposes of this challenge, the knee is considered in three compartments. The medial compartment lies between the medial femoral condyle and the medial tibial plateau, on the inner side of the knee. The lateral compartment lies between the lateral femoral condyle and the lateral tibial plateau, on the outer side. The patellofemoral compartment lies between the patella and the femoral trochlea, the groove on the front of the femur. Osteoarthritis, or cartilage loss, is assessed separately in each of these three compartments.

Four ligaments stabilize the knee. The anterior cruciate ligament (ACL) and posterior cruciate ligament (PCL) lie within the joint, in the intercondylar notch, and control front-to-back stability and rotation. The medial collateral ligament (MCL) and lateral collateral ligament (LCL) run along the inner and outer sides of the knee and resist side-to-side stress. This challenge focuses on tears of the ACL and the MCL.

Between the femur and tibia sit two menisci, the medial meniscus and the lateral meniscus, C-shaped wedges of fibrocartilage that cushion the joint, absorb shock, and improve the fit between the rounded femur and the relatively flat tibia. Tears of the medial and lateral meniscus are evaluated separately.

Other relevant structures include the bone marrow within the femur, tibia, and patella, where a bruise from impact is called a bone contusion (or bone marrow edema) and a break in the bone is a fracture; and the soft tissues behind the knee, where a fluid-filled outpouching of the joint lining is called a Baker, or popliteal, cyst.

## Imaging Overview

Several modalities can image the knee. Radiographs (X-rays) show bones and joint-space narrowing but cannot directly visualize ligaments, menisci, or cartilage. Ultrasound and CT have specific roles but offer limited soft-tissue contrast for the internal structures of the joint. Magnetic resonance imaging (MRI) is the reference standard for evaluating internal derangement of the knee because it provides excellent soft-tissue contrast, depicts ligaments, menisci, cartilage, and bone marrow simultaneously, and does not use ionizing radiation.

A knee MRI examination is not a single image but a set of series, each acquired with a particular pulse sequence and in a particular imaging plane. The three standard planes are axial (cross-sectional, viewed as if looking up through the leg), coronal (a front-facing, side-to-side view), and sagittal (a side-profile, front-to-back view). Different structures are best seen in different planes; for example, the cruciate ligaments and menisci are well evaluated on sagittal and coronal images, and the patellofemoral cartilage on axial images.

Sequences differ in how they weight tissue signal. Fluid-sensitive sequences, such as proton-density or T2-weighted images, often with fat suppression, make edema, effusion, and tears appear bright and are central to detecting most abnormalities; a meniscal tear, for instance, appears as abnormally increased signal that reaches the surface of the meniscus on more than one image. Other sequences emphasize anatomy or cartilage detail. Interpreting a study therefore requires integrating information across multiple series, planes, and slices for the same knee, an inherently three-dimensional task.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2F055d8dc92fe53e98325b81b9eae26dee%2Fimage3.jpg?generation=1786032046408361&alt=media)

*Sagittal proton-density, fat-suppressed (fluid-sensitive) MRI of the knee, the type of sequence on which most abnormalities are detected. Image by Ptrump16, CC BY-SA 4.0, via Wikimedia Commons.*

For the purpose of the 2026 Knee Abnormality Challenge, a “fluid sensitive” sequence refers to one in which edema, hemorrhage, and other types of fluid appear bright and fat is suppressed in some way.

Because findings can be subtle and the criteria for what counts as clinically significant are nuanced, interpretation varies between readers. This variability, combined with limited access to subspecialty MSK radiologists, motivates the development of consistent automated tools.

## Label Description

Models are evaluated on twelve binary labels, each indicating the presence or absence of a specific finding in the imaged knee. The labels, and the criteria used by the annotating radiologists, are summarized below. In each case, ambiguous or borderline findings (“on the fence”) were graded as negative to favor specificity.

ACL tear: A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted, with or without secondary signs such as characteristic pivot-shift bone contusions. Mild signal change, degeneration, or thickening without discontinuity is graded negative.

MCL tear: A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative.

Medial meniscus tear: Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.

Lateral meniscus tear: The same criteria applied to the lateral meniscus.

Medial compartment osteoarthritis: A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the medial compartment, with or without underlying subchondral marrow changes.

Lateral compartment osteoarthritis: The same criteria applied to the lateral compartment.

Patellofemoral compartment osteoarthritis: The same criteria applied to the patellofemoral compartment.

Joint effusion: A moderate or large amount of fluid distending the joint.

Synovitis: Inflammation and thickening of the synovial lining of the joint.

Baker (popliteal) cyst: A moderate or large fluid collection in the characteristic location behind the knee.

Contusion: A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line.

Acute fracture: An acute cortical break or fracture line.

Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus ground truth. Labels are assigned at the level of the whole examination, for a single knee.

## Examples

The following annotated examples, drawn from the challenge’s image annotation guidelines, illustrate representative positive and negative cases for several of the labels. In each figure, yellow arrows and arrowheads mark the findings described in the caption.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2Fffa3eef262fb5e7d930af5977835e348%2F1n.png?generation=1786031398436425&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2F72315329db3196042e507da265c5c5a0%2F1p.png?generation=1786031402755514&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2F02e12a96962914c53a984278c8710bd1%2F2n.png?generation=1786031407998401&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2F9b2477692c6cb65dfe412b0e84e4e54c%2F2n2.png?generation=1786031412122917&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2F3ca890a0dd86b59acfaee1dfe80d1f42%2F2p.png?generation=1786031416545147&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2Fbc1bda64b7ddd42dca2c82e38a9ed688%2F2p2.png?generation=1786031421281409&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2F064beb3b6c9c0fd333714640a8ea609b%2F3n%20(2).png?generation=1786031426950658&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2F59aa2791484d8ea557310a7bd8cf06e6%2F3n.png?generation=1786031439268501&alt=media)


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2F0ed8b17ee998a01621ccab7b386c27fa%2F3p.png?generation=1786031524108353&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2F101f250c5b898b919f866d341d6a1520%2F4n.png?generation=1786031536699757&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2Fc0497ed5a69b2851731ed7652ee57e79%2F4n2.png?generation=1786031554221628&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2Ffc254271a446b9c159539c24e07ae202%2F4p.png?generation=1786031624514519&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F413040%2F826e3f821e06a4c58635339f4459c4ee%2F4p2.png?generation=1786031636743915&alt=media)

## Comments (17)

### **?** · *2026-08-08T21:26:43.573000Z*

thank you so much for the detailed anatomical explanations!

### **?** · *2026-08-09T05:44:24.423000Z*

many Thanks !

### **?** · *2026-08-09T20:03:03.927000Z*

Gracias por los conocimientos!

### **?** · 7 votes · *2026-08-10T12:14:23.900000Z*

For the train.csv. It contains only 58 flagged labels. Rest are NaN.
Those flags do not reflect what is the report.
Does the reports are the GT?

#### **?** · *2026-10-10T05:36:17.797000Z*

Yep. You need to convert them to target features. Methodology is totally up to you, LLM, regex etc...

### **?** · 4 votes · *2026-08-10T21:41:22.047000Z*

Thanks for the overview! For anyone getting started, the key insight is that this is a multi-label classification problem — each study can have multiple abnormalities present simultaneously. The 12 abnormality categories each need their own probability prediction.

### **?** · *2026-08-18T09:20:21.520000Z*

Great overview. The combination of multi-label MRI classification, multilingual reports, and clinically adjudicated labels makes this a particularly interesting multimodal challenge. Has anyone found the report text to add a clear advantage over image-only models?

### **?** · *2026-09-03T06:53:18.723000Z*

Hi @javaduke95 - thank you offering such clear explanation to competitors. can i clarify the below in the label description section: should "the same criteria applied" read medial meniscus rather than lateral meniscus?

> Medial meniscus tear: Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.
>
> Lateral meniscus tear: The same criteria applied to the lateral meniscus.

### **?** · *2026-09-03T21:34:49.027000Z*

this details are very important. Thank you

### **?** · *2026-10-07T05:17:51.717000Z*

Hi study organizers: can you comment on how to deal with bilateral knee mri's in which both left and right side are within the same studyid? If there is only one label per studyid then how should this be resolved? Just report every positive in either knee? For example if left knee ACL is torn but right knee ACL is normal then how should it be labeled if both studies are within same studyid?

#### **?** · *2026-10-11T08:16:36.580000Z*

Yes, that would be great to know. 7 studies with both knees covered are in the training data, but I wonder if the official test data also contain such studies?

### **?** · *2026-10-12T05:49:54.863000Z*

When the winner announced

### **?** · *2026-08-08T16:36:50.503000Z*

Many thanks

### **?** · *2026-08-09T18:36:30.903000Z*

Thank you for sharing this knowledge.

### **?** · *2026-09-11T09:10:34.307000Z*

Thank you for the details.

### **?** · *2026-09-11T16:24:51.620000Z*

many Thanks !
