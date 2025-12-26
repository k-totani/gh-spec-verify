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

// カテゴリ定数
const (
	CategoryAPI = "api"
	CategoryUI  = "ui"
)

// ExtractOptions はエンドポイント抽出時のオプション
type ExtractOptions struct {
	// ソースタイプ (express, auto など)
	SourceType string
	// カテゴリ (api, ui)
	Category string
}

// IsUICategory はUIカテゴリかどうかを判定する
func (o *ExtractOptions) IsUICategory() bool {
	return o != nil && o.Category == CategoryUI
}

// GetSourceType はソースタイプを取得する（nilセーフ）
func (o *ExtractOptions) GetSourceType() string {
	if o == nil {
		return ""
	}
	return o.SourceType
}

// VerifyOptions は検証時のオプション
type VerifyOptions struct {
	// 検証観点（AIへのヒント）
	VerificationFocus []string
}

// Provider はAIプロバイダーのインターフェース
type Provider interface {
	// Verify はSPECとコードの一致度を検証する
	Verify(ctx context.Context, specContent string, codeContents map[string]string) (*VerificationResult, error)

	// VerifyWithOptions は検証観点を指定してSPECとコードの一致度を検証する
	VerifyWithOptions(ctx context.Context, specContent string, codeContents map[string]string, opts *VerifyOptions) (*VerificationResult, error)

	// ExtractEndpoints はコードからAPIエンドポイント/ページルートを抽出する
	ExtractEndpoints(ctx context.Context, opts *ExtractOptions, codeContent string) ([]EndpointResult, error)

	// Name はプロバイダー名を返す
	Name() string
}

// NewProvider は指定されたプロバイダーのインスタンスを作成する
func NewProvider(providerName string, apiKey string) (Provider, error) {
	switch providerName {
	case "claude", "anthropic":
		return NewClaudeProvider(apiKey)
	case "openai", "gpt":
		return NewOpenAIProvider(apiKey)
	case "gemini", "google":
		return NewGeminiProvider(apiKey)
	default:
		return NewClaudeProvider(apiKey)
	}
}
