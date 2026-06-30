import { useState } from 'react'
import { AudioUploader } from './AudioUploader'
import { setVoiceSample, verifySpeaker } from '../api/speaker'
import './SpeakerVerify.css'

type RefState = 'default' | 'setting' | 'set'
type VerifyState = 'idle' | 'loading' | 'done' | 'error'

export function SpeakerVerify() {
  const [refFile, setRefFile] = useState<File | null>(null)
  const [refState, setRefState] = useState<RefState>('default')

  const [verifyFile, setVerifyFile] = useState<File | null>(null)
  const [verifyState, setVerifyState] = useState<VerifyState>('idle')
  const [sameSpeaker, setSameSpeaker] = useState<boolean | null>(null)
  const [confidence, setConfidence] = useState<number | null>(null)
  const [error, setError] = useState('')

  const handleSetRef = async () => {
    if (!refFile) return
    setRefState('setting')
    try {
      await setVoiceSample(refFile)
      setRefState('set')
    } catch {
      setRefState('default')
    }
  }

  const handleVerify = async () => {
    if (!verifyFile) return
    setVerifyState('loading')
    setSameSpeaker(null)
    setConfidence(null)
    try {
      const res = await verifySpeaker(verifyFile)
      setSameSpeaker(res.same_speaker)
      setConfidence(res.confidence ?? null)
      setVerifyState('done')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong.')
      setVerifyState('error')
    }
  }

  const refSubLabel =
    refState === 'set' && refFile
      ? refFile.name
      : 'Default sample active on server'

  return (
    <>
      <section className="hero">
        <h1 className="hero-title">
          Confirm the<br />
          <span className="hero-accent">voice.</span>
        </h1>
        <p className="hero-sub">
          Set a reference recording, then check if a new clip belongs to the same speaker.
        </p>
      </section>

      <section className="content">

        {/* ── Step 1: Reference ── */}
        <div className="sv-step">
          <div className="sv-step-header">
            <span className="sv-step-num">01</span>
            <div className="sv-step-info">
              <p className="sv-step-title">Reference Voice</p>
              <p className="sv-step-sub" title={refSubLabel}>{refSubLabel}</p>
            </div>
            {refState === 'set'
              ? <span className="sv-badge sv-badge--ok">Set ✓</span>
              : <span className="sv-badge sv-badge--default">Default</span>
            }
          </div>

          <AudioUploader
            onFile={f => { setRefFile(f); if (refState === 'set') setRefState('default') }}
            selected={refFile}
            active={refState === 'setting'}
          />

          {refFile && refState !== 'setting' && refState !== 'set' && (
            <button className="btn-primary" onClick={handleSetRef}>
              Set as Reference
            </button>
          )}

          {refState === 'setting' && (
            <div className="status-row">
              <span className="spinner" />
              <span>Uploading reference…</span>
            </div>
          )}
        </div>

        {/* ── Step 2: Verify ── */}
        <div className="sv-step">
          <div className="sv-step-header">
            <span className="sv-step-num">02</span>
            <div className="sv-step-info">
              <p className="sv-step-title">Verify Sample</p>
              <p className="sv-step-sub">Upload audio to compare against the reference</p>
            </div>
          </div>

          <AudioUploader
            onFile={f => { setVerifyFile(f); setVerifyState('idle') }}
            selected={verifyFile}
            active={verifyState === 'loading'}
          />

          {verifyFile && verifyState !== 'loading' && (
            <button className="btn-primary" onClick={handleVerify}>
              Verify Speaker
            </button>
          )}

          {verifyState === 'loading' && (
            <div className="status-row">
              <span className="spinner" />
              <span>Comparing voices…</span>
            </div>
          )}

          {verifyState === 'error' && (
            <div className="result-error">{error}</div>
          )}
        </div>

        {/* ── Result ── */}
        {verifyState === 'done' && sameSpeaker !== null && (
          <div className={`sv-result sv-result--${sameSpeaker ? 'match' : 'nomatch'}`}>
            <div className="sv-result-icon">
              {sameSpeaker ? '✓' : '✗'}
            </div>
            <p className="sv-result-title">
              {sameSpeaker ? 'Same Speaker' : 'Different Speaker'}
            </p>
            <p className="sv-result-sub">
              {sameSpeaker
                ? 'The voice matches the reference recording.'
                : 'The voice does not match the reference recording.'}
            </p>
            {confidence !== null && (
              <div className="result-confidence sv-result-confidence">
                <span className="result-confidence__label">Confidence</span>
                <div className="result-confidence__track">
                  <div
                    className="result-confidence__fill"
                    style={{
                      width: `${Math.round(confidence * 100)}%`,
                      background: sameSpeaker ? 'var(--success)' : 'var(--error)',
                    }}
                  />
                </div>
                <span className="result-confidence__value">
                  {Math.round(confidence * 100)}%
                </span>
              </div>
            )}
          </div>
        )}

      </section>
    </>
  )
}