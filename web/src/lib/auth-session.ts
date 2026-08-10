"use client";

import { login } from "@/lib/api";
import { clearStoredAuthSession, getStoredAuthSession, setStoredAuthSession, type StoredAuthSession } from "@/store/auth";

export async function getValidatedAuthSession(): Promise<StoredAuthSession | null> {
  const storedSession = await getStoredAuthSession();
  if (!storedSession) {
    return null;
  }

  try {
    const data = await login(storedSession.key);
    const nextSession: StoredAuthSession = {
      key: storedSession.key,
      role: data.role,
      subjectId: data.subject_id,
      name: data.name,
      dailyRequestLimit: data.daily_request_limit,
      dailyRequestUsed: data.daily_request_used,
      dailyRequestRemaining: data.daily_request_remaining,
      dailyRequestDate: data.daily_request_date || "",
      imageRequestLimit: data.image_request_limit,
    };
    await setStoredAuthSession(nextSession);
    return nextSession;
  } catch {
    await clearStoredAuthSession();
    return null;
  }
}
