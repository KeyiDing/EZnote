# EZnote

## Inspiration 💭
Have you ever imagined how hard it is for students with disability to take notes during lectures? 
According to data from the National Center for Education Statistics, there are nearly 7 million disabled students in the U.S., which makes up 14% of national public school enrollment. For many of them, note-taking is an incredibly difficult task. To tackle this problem, some schools like our school, Johns Hopkins University, offer note-taking accommodations by recruiting peer note-takers. However, this approach is sometimes ineffective as there is often a lack of volunteering note-takers, and it also results in a waste of resources.

## What it does ✨
EZnote is an online automatic note-taking tool designed to enhance the learning experience for students with disability. Students can choose to upload a lecture recording or film a lecture in real-time, and EZnote will automatically generate the notes in PDF format with great efficiency and precision. 

Specifically, EZnote uses Computer Vision and Machine Learning to recognize the writing board and the teacher in the lecture video. As the lecture video is being analyzed, everything written on the board by the teacher, including graphs and special symbols, can be captured and generated as PDF files. Moreover, we have written algorithms to "remove" the teacher from the background when the teacher is standing in front of the writing board, effectively improving the accuracy.

## How we built it 🧰
EZnote is built with TensorFlow and OpenCV libraries in python. The application backend is built with Node.js and the frontend is built with JavaScript, HTML, and CSS.

## How to use it 🤔
* Open client.html.
* Select a lecture you wish to be converted into notes.
* Click upload.
* EZnote automatically generates notes in pdf version for you to download!

## Challenges we ran into 💪
* The human recognition model does not always track the teacher. Sometimes, it gets "distracted" by other people or objects in the image.
* The text extraction ability is limited by the resolution of the camera.
* Coordinates transformation is difficult to implement under different frames.
* Building frontend and backend is more difficult than we thought - we took a long time learning how to build file uploading and processing system and integrate client side with server side.

## Accomplishments that we're proud of 😊
We have achieved all the main functionalities of EZnote, such as
* automatic recognition of teacher, writing board, and texts in the images
* successfully "removal" of teacher from the images while keeping the background pixels
* minimizing noise introduced during image processing
* optimizing running time to ensure the code works in time series and gives real-time feedback

## What we learned 📚
* computer vision algorithms and image processing
* supervised machine learning
* web development

## What's next for EZnote 🌐
We have many things in mind for EZnote. 
* We will enable EZnote to generate notes in real-time as the lecture video is being recorded. This can be done by comparing the difference in matrices (which are calculated from screenshots of the writing board) at different moments. 
* We will also use a similar method to train EZnote to distinguish minor correction from full board erasing. 
* We will make EZnote more scalable and production ready to benefit all students with disability across the world.
