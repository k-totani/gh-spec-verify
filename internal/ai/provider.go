package ai

import (
	"context"
)

// VerificationResult は検証結果を表す
type VerificationResult struct {
	// 一致度（0-100）
	MatchPercentage int `json:"matchPercentage"`

	// 一致している項目
	MatchedItems []string `json:"matchedItems"`

	// 一致していない項目
	UnmatchedItems []string `json:"unmatchedItems"`

	// 補足コメント
	Notes string `json:"notes"`
}

// EndpointResult はエンドポイント抽出結果を表す
type EndpointResult struct {
	Method      string `json:"method"`
	Path        string `json:"path"`
	Source      string `json:"source,omitempty"`
	File        string `json:"file,omitempty"`
	Description string `json:"description,omitempty"`
}

// Provider はAIプロバイダーのインターフェース
type Provider interface {
	// Verify はSPECとコードの一致度を検証する
	Verify(ctx context.Context, specContent string, codeContents map[string]string) (*VerificationResult, error)

	// ExtractEndpoints はコードからAPIエンドポイントを抽出する
	ExtractEndpoints(ctx context.Context, sourceType string, codeContent string) ([]EndpointResult, error)

	// Name はプロバイダー名を返す
	Name() string
}

// NewProvider は指定されたプロバイダーのインスタンスを作成する
func NewProvider(providerName string, apiKey string) (Provider, error) {
	switch providerName {
	case "claude", "anthropic":
		return NewClaudeProvider(apiKey)
	// 将来的に他のプロバイダーを追加
	// case "openai":
	// 	return NewOpenAIProvider(apiKey)
	// case "gemini":
	// 	return NewGeminiProvider(apiKey)
	default:
		return NewClaudeProvider(apiKey)
	}
}
