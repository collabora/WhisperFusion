class AudioStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSize = 4096;
    this.buffer = new Float32Array(this.chunkSize);
    this.bufferPointer = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    let channelCount = Math.min(input.length, output.length);

    for (let i = 0; i < input[0].length; i++) {
      this.buffer[this.bufferPointer++] = input[0][i];

      if (this.bufferPointer >= this.chunkSize) {
        this.port.postMessage(this.buffer);
        this.bufferPointer = 0;
      }
    }

    for (let channel = 0; channel < channelCount; ++channel) {
      output[channel].set(input[channel]);
    }

    return true;
  }
}

registerProcessor("audio-stream-processor", AudioStreamProcessor);
