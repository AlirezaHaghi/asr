import { useState } from 'react'
import { AudioUploader } from './components/AudioUploader'
import { transcribeAudio } from './api/transcribe'
import './App.css'

type Status = 'idle' | 'uploading' | 'done' | 'error'

export default function App() {
  const [status, setStatus] = useState<Status>('idle')
  const [file, setFile] = useState<File | null>(null)
  const [transcription, setTranscription] = useState('')
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)

  const handleFile = (f: File) => {
    setFile(f)
    setStatus('idle')
    setTranscription('')
    setError('')
  }

  const handleTranscribe = async () => {
    if (!file) return
    setStatus('uploading')
    try {
      const result = await transcribeAudio(file)
      setTranscription(result.transcription)
      setStatus('done')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
      setStatus('error')
    }
  }

  const handleCopy = async () => {
    await navigator.clipboard.writeText(transcription)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <main className="layout">
      <header className="nav">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-name">Transcribe</span>
        </div>
      </header>

      <section className="hero">
        <h1 className="hero-title">
          Audio to text,<br />
          <span className="hero-accent">instantly.</span>
        </h1>
        <p className="hero-sub">
          Upload an audio file and receive an accurate transcription in seconds.
        </p>
      </section>

      <section className="content">
        <AudioUploader
          onFile={handleFile}
          selected={file}
          active={status === 'uploading'}
        />

        {file && status !== 'uploading' && (
          <button className="btn-primary" onClick={handleTranscribe}>
            Transcribe
          </button>
        )}

        {status === 'uploading' && (
          <div className="status-row">
            <span className="spinner" />
            <span>Processing audio…</span>
          </div>
        )}

        {status === 'error' && (
          <div className="result-error">{error}</div>
        )}

        {status === 'done' && transcription && (
          <div className="result">
            <div className="result-header">
              <span className="result-label">Transcription</span>
              <button className="btn-copy" onClick={handleCopy}>
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
            <p className="result-body">{transcription}</p>
          </div>
        )}
      </section>
    </main>
  )
}