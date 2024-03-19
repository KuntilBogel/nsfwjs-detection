(async function () {
    require('dotenv').config()
    const express = require('express')
    const tf = require("@tensorflow/tfjs-node")
    const sharp = require("sharp");
    const jpeg = require("jpeg-js")
    const ffmpeg = require("fluent-ffmpeg")
    const { fileTypeFromBuffer } = (await import('file-type'));
    const stream = require("stream")
    const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
    const ffprobePath = require('@ffprobe-installer/ffprobe').path;
    const nsfwjs = require("nsfwjs");
    const fs = require("fs")
    ffmpeg.setFfprobePath(ffprobePath);
    ffmpeg.setFfmpegPath(ffmpegPath);
    // require("./model").loadModel()
    const app = express()
    const model = await nsfwjs.load("InceptionV3");
    app.use(express.json())

    app.all('/', async (req, res) => {
        try {
            const { img, auth } = req.query
            if (img) {
                if (process.env.AUTH) {
                    if (!auth || process.env.AUTH != auth) return res.send("Invalid auth code")
                }
                const imageBuffer = await fetch(img).then(async c => await c.arrayBuffer())
                // console.log((await fileTypeFromBuffer(imageBuffer)).mime)
                if ((await fileTypeFromBuffer(imageBuffer)).mime.includes("image")) {
                    const convertedBuffer = await sharp(Buffer.from(imageBuffer)).jpeg().toBuffer(); // convert webp to jpeg
                    const image = await convert(convertedBuffer)
                    const predictions = await model.classify(image);
                    image.dispose(); // Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).
                    return res.send(predictions);
                } else {
                    let inputStream1 = new stream.PassThrough();
                    inputStream1.end(Buffer.from(imageBuffer));

                    ffmpeg.ffprobe(inputStream1, function (err, metadata) {
                        if (err) {
                            console.error(err);
                            return;
                        }

                        // Get a random second
                        const randomSecond = Math.floor(Math.random() * metadata.format.duration);

                        // Create a new input stream for the ffmpeg command
                        let inputStream2 = new stream.PassThrough();
                        inputStream2.end(Buffer.from(imageBuffer));

                        // Create a PassThrough stream to collect the output
                        const output = new stream.PassThrough();

                        // Set up the ffmpeg command
                        ffmpeg({ source: inputStream2 })
                            .seekInput(randomSecond)
                            .outputOptions('-vframes', '1')
                            .outputOptions('-f', 'image2pipe')
                            .outputOptions('-vcodec', 'png')
                            .output(output)
                            .on('error', console.error)
                            .run();

                        // Collect the output into a buffer
                        const chunks = [];
                        output.on('data', chunk => chunks.push(chunk));
                        output.on('end', async () => {
                            const buffer = Buffer.concat(chunks);
                            fs.writeFileSync("aa.png", buffer)
                            const convertedBuffer = await sharp(buffer).jpeg().toBuffer(); // convert webp to jpeg
                            const cimage = await convert(convertedBuffer)
                            const apredictions = await model.classify(cimage);
                            cimage.dispose(); // Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).
                            return res.send(apredictions);
                        });
                    });
                }

            }else{
                return res.send('Hello World!')
            }
        } catch (err) {
            console.log(err)
            return res.status(500).json({ error: err.toString() })
        }
    })

    const port = process.env.PORT || process.env.SERVER_PORT || 7860

    app.listen(port, () => {
        console.log(`Example app listening on port ${port}`)
    })
    const convert = async (img) => {
        // Decoded image in UInt8 Byte array
        const image = await jpeg.decode(img, { useTArray: true });
        const numChannels = 3;
        const numPixels = image.width * image.height;
        const values = new Int32Array(numPixels * numChannels);
        for (let i = 0; i < numPixels; i++)
            for (let c = 0; c < numChannels; ++c)
                values[i * numChannels + c] = image.data[i * 4 + c];
        return tf.tensor3d(values, [image.height, image.width, numChannels], "int32");
    };
})()