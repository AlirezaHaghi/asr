import { useState } from 'react'
import { AudioUploader } from './components/AudioUploader'
import { SpeakerVerify } from './components/SpeakerVerify'
import { transcribeAudio } from './api/transcribe'
import './App.css'

type Status = 'idle' | 'uploading' | 'done' | 'error'
type Tab = 'transcribe' | 'verify'

export default function App() {
  const [tab, setTab] = useState<Tab>('transcribe')

  const [status, setStatus] = useState<Status>('idle')
  const [file, setFile] = useState<File | null>(null)
  const [transcription, setTranscription] = useState('')
  const [accent, setAccent] = useState<string | undefined>(undefined)
  const [confidence, setConfidence] = useState<number | undefined>(undefined)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)

  const handleFile = (f: File) => {
    setFile(f)
    setStatus('idle')
    setTranscription('')
    setAccent(undefined)
    setConfidence(undefined)
    setError('')
  }

  const handleTranscribe = async () => {
    if (!file) return
    setStatus('uploading')
    try {
      const result = await transcribeAudio(file)
      setTranscription(result.transcription)
      setAccent(result.accent)
      setConfidence(result.confidence)
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

  const hasMeta = accent != null || confidence != null

  return (
    <main className="layout">
      <header className="nav">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-name">Transcribe</span>
        </div>
        <div className="tabs">
          <button
            className={`tab${tab === 'transcribe' ? ' tab--active' : ''}`}
            onClick={() => setTab('transcribe')}
          >
            Transcribe
          </button>
          <button
            className={`tab${tab === 'verify' ? ' tab--active' : ''}`}
            onClick={() => setTab('verify')}
          >
            Verify Speaker
          </button>
        </div>
      </header>

      {tab === 'verify' ? (
        <SpeakerVerify />
      ) : (
        <>
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
                {hasMeta && (
                  <div className="result-meta">
                    {accent != null && (
                      <span className="result-tag">{accent}</span>
                    )}
                    {confidence != null && (
                      <div className="result-confidence">
                        <span className="result-confidence__label">Confidence</span>
                        <div className="result-confidence__track">
                          <div
                            className="result-confidence__fill"
                            style={{ width: `${Math.round(confidence * 100)}%` }}
                          />
                        </div>
                        <span className="result-confidence__value">
                          {Math.round(confidence * 100)}%
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </section>
        </>
      )}
    </main>
  )
}