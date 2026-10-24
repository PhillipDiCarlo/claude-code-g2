import type { Request, Response } from 'express'
import { spawn } from 'node:child_process'
import * as fs from 'node:fs/promises'
import * as os from 'node:os'
import * as path from 'node:path'
import * as crypto from 'node:crypto'

// POST /api/transcribe. Accepts `audio/wav` or `audio/pcm` (raw 16 kHz s16le
// mono — the glasses format). Raw PCM gets a 44-byte RIFF header prepended
// so whisper.cpp accepts it without an ffmpeg hop.
//
// Transcription runs locally via the whisper.cpp CLI (`brew install
// whisper-cpp`). Override the binary / model with WHISPER_BIN and
// WHISPER_MODEL.

const SAMPLE_RATE = 16_000
const CHANNELS = 1
const BITS_PER_SAMPLE = 16

const DEFAULT_WHISPER_MODEL = path.join(os.homedir(), '.cc-g2', 'models', 'ggml-base.en.bin')

function whisperBin(): string {
  return process.env.WHISPER_BIN || 'whisper-cli'
}

function whisperModel(): string {
  return process.env.WHISPER_MODEL || DEFAULT_WHISPER_MODEL
}

export function pcmToWav(
  pcm: Buffer,
  sampleRate = SAMPLE_RATE,
  channels = CHANNELS,
  bitsPerSample = BITS_PER_SAMPLE,
): Buffer {
  const byteRate = (sampleRate * channels * bitsPerSample) / 8
  const blockAlign = (channels * bitsPerSample) / 8
  const dataSize = pcm.length
  const chunkSize = 36 + dataSize

  const header = Buffer.alloc(44)
  header.write('RIFF', 0)
  header.writeUInt32LE(chunkSize, 4)
  header.write('WAVE', 8)
  header.write('fmt ', 12)
  header.writeUInt32LE(16, 16) // subchunk1 size
  header.writeUInt16LE(1, 20)  // PCM
  header.writeUInt16LE(channels, 22)
  header.writeUInt32LE(sampleRate, 24)
  header.writeUInt32LE(byteRate, 28)
  header.writeUInt16LE(blockAlign, 32)
  header.writeUInt16LE(bitsPerSample, 34)
  header.write('data', 36)
  header.writeUInt32LE(dataSize, 40)

  return Buffer.concat([header, pcm])
}

function runWhisper(wavPath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    // -nt: no timestamps, -np: no progress/system info, so stdout is just the text.
    const proc = spawn(whisperBin(), ['-m', whisperModel(), '-f', wavPath, '-l', 'en', '-nt', '-np'], {
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    proc.stdout.on('data', (d) => (stdout += d))
    proc.stderr.on('data', (d) => (stderr += d))
    proc.on('error', reject)
    proc.on('close', (code) => {
      if (code === 0) resolve(stdout)
      else reject(new Error(`${whisperBin()} exited with code ${code}: ${stderr.trim()}`))
    })
  })
}

// whisper.cpp emits one line per segment and bracketed tags like
// [BLANK_AUDIO] for silence; collapse to a single plain string.
function cleanTranscript(raw: string): string {
  return raw
    .replace(/\[[A-Z_ ]+\]/g, '')
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .join(' ')
    .trim()
}

export async function transcribeHandler(req: Request, res: Response): Promise<void> {
  const body = req.body
  if (!Buffer.isBuffer(body) || body.length === 0) {
    res.status(400).json({ error: 'empty audio body' })
    return
  }

  const contentType = (req.header('content-type') ?? '').toLowerCase().split(';')[0]!.trim()
  let wavBuffer: Buffer
  if (contentType === 'audio/wav' || contentType === 'audio/wave' || contentType === 'audio/x-wav') {
    wavBuffer = body
  } else if (contentType === 'audio/pcm' || contentType === 'application/octet-stream' || contentType === '') {
    // Glasses mic audio is raw PCM. Wrap in a WAV header.
    wavBuffer = pcmToWav(body)
  } else {
    res.status(400).json({ error: `unsupported content-type: ${contentType}` })
    return
  }

  const wavPath = path.join(os.tmpdir(), `cc-g2-${crypto.randomUUID()}.wav`)
  try {
    await fs.writeFile(wavPath, wavBuffer)
    const text = cleanTranscript(await runWhisper(wavPath))
    res.json({ text })
  } catch (err) {
    console.error('[transcribe] whisper error:', err)
    res.status(502).json({ error: 'transcription_failed' })
  } finally {
    await fs.rm(wavPath, { force: true })
  }
}
