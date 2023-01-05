class Model {
  constructor(options) {
    this._isModelReady = false;
    this._lastTime = -1;

    this._options = options;
    this._video = options.videoElement;
    this._canvas = options.canvasElement;

    this.initializeModel();
  }

  async initializeModel() {
    ml5.objectDetector("cocossd", {})
      .then(model => {
        this._isModelReady = true;
        this._detector = model;
        console.log("ready");
      })
  }

  detectFrame() {
    if (!this._isModelReady) {
      return;
    }

    let self = this;
    this._detector.detect(this._video, (err, results) => {
      if (err) {
        console.log(err);
        return;
      }

      const ctx = self._canvas.getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      // Font options.
      const font = "16px sans-serif";
      ctx.font = font;
      ctx.textBaseline = "top";
      results.forEach((object) => {
        console.log(object);
        const x = object.x;
        const y = object.y;
        const width = object.width;
        const height = object.height;
        const label = `${object.label}: ${object.confidence.toFixed(2)}`;

        // Draw the bounding box.
        ctx.strokeStyle = "#FFFF3F";
        ctx.lineWidth = 5;
        ctx.strokeRect(x, y, width, height);

        // Draw the label background.
        ctx.fillStyle = "#FFFF3F";
        const textWidth = ctx.measureText(label).width;
        const textHeight = parseInt(font, 10); // base 10
        ctx.fillRect(x, y, textWidth + 4, textHeight + 4);

        ctx.fillStyle = "#000000";
        ctx.fillText(label, x, y);
      });
    });
  }

  start() {
    let self = this;
    this._video.addEventListener("timeupdate", () => {
      self.detectFrame()
    });
  }
}