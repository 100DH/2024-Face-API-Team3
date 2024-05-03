const imageUpload = document.getElementById('imageUpload')

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('./models')
]).then(start)

async function start() {
    // Canvas container 생성
    const container = document.createElement('div')
    container.style.position = 'relative'
    document.body.append(container)

    // 얼굴과 라벨을 매칭한다.
    const labeledFaceDescriptors = await loadLabeledImage()
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
    let image
    let canvas
    document.body.append('준비완료')

    imageUpload.addEventListener('change', async () => {
        if (image) image.remove()
        if (canvas) canvas.remove()

        // 사진을 화면에 표시함
        image = await faceapi.bufferToImage(imageUpload.files[0])

        // 이미지 크기 제한
        if (image.width > 700 || image.height > 800) {
           
            // 이미지 크기를 조정
            image = await resizeImage(image, 700, 800)
        }

        container.append(image)

        // canvas를 초기화 한다
        canvas = faceapi.createCanvasFromMedia(image)
        container.append(canvas)
        const displaySize = {width: image.width, height: image.height}
        faceapi.matchDimensions(canvas, displaySize)

        // 사진에서 얼굴을 식별한다.
        const detections = await faceapi.detectAllFaces(image)
                                .withFaceLandmarks()
                                .withFaceDescriptors()
        
        // 사진에서 얼굴 좌표에 box를 그린다.
        const resizedDetections = faceapi.resizeResults(detections, displaySize)

        // 얼굴에 라벨을 표시한다.
        const result = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
        result.forEach((result, i) => {
            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, {label: result.toString(), boxColor: 'red'})
            drawBox.draw(canvas)
        })
    })
}

// 이미지 크기 조정 함수
async function resizeImage(image, maxWidth, maxHeight) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // 가로 세로 비율을 유지하면서 최대 크기에 맞게 이미지 크기 조정
    const ratio = Math.min(maxWidth / image.width, maxHeight / image.height);
    canvas.width = image.width * ratio;
    canvas.height = image.height * ratio;

    // 이미지 그리기
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    // 이미지를 data URL로 반환
    return new Promise(resolve => {
        canvas.toBlob(blob => {
            const newImage = new Image();
            newImage.src = URL.createObjectURL(blob);
            newImage.onload = () => resolve(newImage);
        });
    });
}

// 얼굴 기초데이터를 초기화 한다.
function loadLabeledImage() {
    const labels = ['박신원', '백대현', '천예준', '최우진']
    return Promise.all(
        labels.map(async label => {
            const description = []
            const img = await faceapi.fetchImage('known/' + label + '.jpg')
            const detections = await faceapi.detectSingleFace(img)
                                .withFaceLandmarks()
                                .withFaceDescriptor()
            description.push(detections.descriptor)
            return new faceapi.LabeledFaceDescriptors(label, description)
        })
    )
}
