const API_BASE = 'http://localhost:8000'

export interface SetVoiceResult {
  reference_path: string
}

export interface VerifyResult {
  same_speaker: boolean
  confidence: number | null
  similarity: number | null
  threshold: number | null
}

export async function setVoiceSample(file: File): Promise<SetVoiceResult> {
  const form = new FormData()
  form.append('audio', file)

  const res = await fetch(`${API_BASE}/set-voice-sample`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export async function verifySpeaker(file: File): Promise<VerifyResult> {
  const form = new FormData()
  form.append('audio', file)

  const res = await fetch(`${API_BASE}/verify-speaker`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}
