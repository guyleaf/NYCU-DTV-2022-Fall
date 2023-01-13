class HlsTech extends EventTarget {
  static Events = {
    MANIFEST_LOADED: "MANIFEST_LOADED",
    LEVEL_SWITCHED: "LEVEL_SWITCHED"
  }

  constructor(_options) {
    super();
    let self = this;
    this._options = _options;
    this._video = _options.videoElement;
    this._config = {
      debug: false,
      enableWorker: true,
      autoStartLoad: true,
      liveSyncDurationCount: 1,
      liveMaxLatencyDurationCount: 5,
      maxLiveSyncPlaybackRate: 2.0,
      liveDurationInfinity: true,
      lowLatencyMode: true,
    };

    this._player = new Hls(this._config);

    this._player.on(Hls.Events.MANIFEST_LOADED, function (event, data) {
      self.dispatchEvent(new CustomEvent(HlsTech.Events.MANIFEST_LOADED, {
        detail: {
          levels: self.getLevels()
        }
      }));
    });

    this._player.on(Hls.Events.LEVEL_SWITCHED, function (event, data) {
      self.dispatchEvent(new CustomEvent(HlsTech.Events.LEVEL_SWITCHED, {
        detail: {
          level: self.getLevel(data.level)
        }
      }));
    })

    this._player.on(Hls.Events.ERROR, function (eventName, data) {
      console.warn('Error event:', data);

      if (data.fatal) {
        switch (data.type) {
          case Hls.ErrorTypes.NETWORK_ERROR:
            // try to recover network error
            console.log('fatal network error encountered, try to recover');
            self.startLoad();
            break;
          case Hls.ErrorTypes.MEDIA_ERROR:
            console.log('fatal media error encountered, try to recover');
            self._player.recoverMediaError();
            self._player.swapAudioCodec();
            break;
          default:
            console.error("Unrecoverable error");
            self.destroy();
            break;
        }
      }
    });
  }

  on(event, callback) {
    this.addEventListener(event, callback, false);
  }

  getLevel(index) {
    index = parseInt(index);
    let level = this._player.levels[index];
    let data = {
      index: level.level != undefined ? level.level : index,
      bitrate: level.bitrate,
      height: level.height,
      width: level.width,
      resolution: level.attrs.RESOLUTION,
      bane: level.name
    };
    return data;
  }

  getLevels() {
    var levels = [];

    for (var i = 0; i < this._player.levels.length; i++) {
      levels.push(this.getLevel(i));
    }

    return levels;
  };

  setLevel(index) {
    index = parseInt(index);
    this._player.nextLevel = index;
  };

  setMaxLevel() {
    var levels = this.getLevels();
    maxQualityIndex = -1;
    bitrate = 0;

    for (var i = 0; i < levels.length; i++) {
      if (levels[i].bitrate > bitrate) {
        bitrate = levels[i].bitrate;
        maxQualityIndex = i;
      }
    }

    this.setLevel(maxQualityIndex);
  };

  start() {
    this._player.loadSource(this._options.src);
    this._player.attachMedia(this._video);
  }

  startLoad() {
    this._player.startLoad(-1);
  }

  stopLoad() {
    this._player.stopLoad();
  }

  reload() {
    this.stopLoad();
    this.startLoad();
  }

  destroy() {
    if (this._player != null) {
      this._player.destroy();
      this._player = null;
    }
  };
}
