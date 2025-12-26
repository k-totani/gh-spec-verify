package ai

import (
	"testing"
)

func TestNewProvider(t *testing.T) {
	tests := []struct {
		name         string
		providerName string
		apiKey       string
		wantName     string
		wantErr      bool
	}{
		{
			name:         "claude provider",
			providerName: "claude",
			apiKey:       "test-key",
			wantName:     "claude",
			wantErr:      false,
		},
		{
			name:         "anthropic alias",
			providerName: "anthropic",
			apiKey:       "test-key",
			wantName:     "claude",
			wantErr:      false,
		},
		{
			name:         "openai provider",
			providerName: "openai",
			apiKey:       "test-key",
			wantName:     "openai",
			wantErr:      false,
		},
		{
			name:         "gpt alias",
			providerName: "gpt",
			apiKey:       "test-key",
			wantName:     "openai",
			wantErr:      false,
		},
		{
			name:         "gemini provider",
			providerName: "gemini",
			apiKey:       "test-key",
			wantName:     "gemini",
			wantErr:      false,
		},
		{
			name:         "google alias",
			providerName: "google",
			apiKey:       "test-key",
			wantName:     "gemini",
			wantErr:      false,
		},
		{
			name:         "default to claude",
			providerName: "unknown",
			apiKey:       "test-key",
			wantName:     "claude",
			wantErr:      false,
		},
		{
			name:         "empty api key",
			providerName: "claude",
			apiKey:       "",
			wantName:     "",
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := NewProvider(tt.providerName, tt.apiKey)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error but got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if provider.Name() != tt.wantName {
				t.Errorf("provider name = %q, want %q", provider.Name(), tt.wantName)
			}
		})
	}
}

func TestNewClaudeProvider(t *testing.T) {
	t.Run("with valid api key", func(t *testing.T) {
		p, err := NewClaudeProvider("test-key")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if p.Name() != "claude" {
			t.Errorf("name = %q, want %q", p.Name(), "claude")
		}
	})

	t.Run("with empty api key", func(t *testing.T) {
		_, err := NewClaudeProvider("")
		if err == nil {
			t.Error("expected error but got nil")
		}
	})
}

func TestNewOpenAIProvider(t *testing.T) {
	t.Run("with valid api key", func(t *testing.T) {
		p, err := NewOpenAIProvider("test-key")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if p.Name() != "openai" {
			t.Errorf("name = %q, want %q", p.Name(), "openai")
		}
	})

	t.Run("with empty api key", func(t *testing.T) {
		_, err := NewOpenAIProvider("")
		if err == nil {
			t.Error("expected error but got nil")
		}
	})
}

func TestNewGeminiProvider(t *testing.T) {
	t.Run("with valid api key", func(t *testing.T) {
		p, err := NewGeminiProvider("test-key")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if p.Name() != "gemini" {
			t.Errorf("name = %q, want %q", p.Name(), "gemini")
		}
	})

	t.Run("with empty api key", func(t *testing.T) {
		_, err := NewGeminiProvider("")
		if err == nil {
			t.Error("expected error but got nil")
		}
	})
}
