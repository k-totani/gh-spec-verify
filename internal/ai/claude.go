package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
)

const claudeAPIURL = "https://api.anthropic.com/v1/messages"

// ClaudeProvider はClaude APIを使用したプロバイダー
type ClaudeProvider struct {
	apiKey string
	model  string
}

// NewClaudeProvider は新しいClaudeProviderを作成する
func NewClaudeProvider(apiKey string) (*ClaudeProvider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	return &ClaudeProvider{
		apiKey: apiKey,
		model:  "claude-sonnet-4-20250514",
	}, nil
}

// Name はプロバイダー名を返す
func (p *ClaudeProvider) Name() string {
	return "claude"
}

// claudeRequest はClaude APIへのリクエスト
type claudeRequest struct {
	Model     string          `json:"model"`
	MaxTokens int             `json:"max_tokens"`
	Messages  []claudeMessage `json:"messages"`
}

type claudeMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// claudeResponse はClaude APIからのレスポンス
type claudeResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Verify はSPECとコードの一致度を検証する
func (p *ClaudeProvider) Verify(ctx context.Context, specContent string, codeContents map[string]string) (*VerificationResult, error) {
	return p.VerifyWithOptions(ctx, specContent, codeContents, nil)
}

// VerifyWithOptions は検証観点を指定してSPECとコードの一致度を検証する
func (p *ClaudeProvider) VerifyWithOptions(ctx context.Context, specContent string, codeContents map[string]string, opts *VerifyOptions) (*VerificationResult, error) {
	var prompt string
	if opts != nil && len(opts.VerificationFocus) > 0 {
		prompt = buildVerificationPromptWithFocus(specContent, codeContents, opts.VerificationFocus)
	} else {
		prompt = buildVerificationPrompt(specContent, codeContents)
	}

	req := claudeRequest{
		Model:     p.model,
		MaxTokens: 2000,
		Messages: []claudeMessage{
			{Role: "user", Content: prompt},
		},
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", claudeAPIURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var claudeResp claudeResponse
	if err := json.Unmarshal(body, &claudeResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if claudeResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", claudeResp.Error.Message)
	}

	if len(claudeResp.Content) == 0 {
		return nil, fmt.Errorf("empty response from API")
	}

	return parseVerificationResult(claudeResp.Content[0].Text)
}

// buildCodeSection はコードセクションを構築する共通関数
func buildCodeSection(codeContents map[string]string) string {
	var codeSection strings.Builder
	for filePath, content := range codeContents {
		codeSection.WriteString(fmt.Sprintf("\n### %s\n```\n%s\n```\n", filePath, content))
	}
	return codeSection.String()
}

// getDefaultVerificationFocus はデフォルトの検証観点を返す
func getDefaultVerificationFocus() []string {
	return []string{
		"画面構成: SPECに記載された要素がコードに存在するか",
		"状態管理: SPECに記載された状態やフックが使用されているか",
		"処理フロー: SPECに記載された処理フローがコードで実装されているか",
		"バリデーション: SPECに記載されたバリデーションルールが実装されているか",
		"エラーハンドリング: SPECに記載されたエラーケースが処理されているか",
	}
}

// buildVerificationPrompt は検証用のプロンプトを構築する
// デフォルトの検証観点を使用してbuildVerificationPromptWithFocusを呼び出す
func buildVerificationPrompt(specContent string, codeContents map[string]string) string {
	return buildVerificationPromptWithFocus(specContent, codeContents, getDefaultVerificationFocus())
}

// buildVerificationPromptWithFocus はカスタム検証観点を含むプロンプトを構築する
func buildVerificationPromptWithFocus(specContent string, codeContents map[string]string, verificationFocus []string) string {
	codeSection := buildCodeSection(codeContents)

	// 検証観点をフォーマット
	var focusSection strings.Builder
	for i, focus := range verificationFocus {
		focusSection.WriteString(fmt.Sprintf("%d. %s\n", i+1, focus))
	}

	return fmt.Sprintf(`あなたはコードレビューの専門家です。以下のSPEC(仕様書)と実際のコードを比較して、一致度を評価してください。

## SPEC(仕様書)
%s

## 実際のコード
%s

## 評価基準
以下の観点で重点的に評価してください:
%s

## 出力形式
以下のJSON形式で出力してください:
%sjson
{
  "matchPercentage": <0-100の数値>,
  "matchedItems": ["一致している項目1", "一致している項目2", ...],
  "unmatchedItems": ["一致していない項目1", "一致していない項目2", ...],
  "notes": "補足コメント(未実装の機能や改善点など)"
}
%s

JSONのみを出力してください。`, specContent, codeSection, focusSection.String(), "```", "```")
}

// parseVerificationResult はClaude APIのレスポンスから検証結果を抽出する
func parseVerificationResult(text string) (*VerificationResult, error) {
	// JSONブロックを抽出
	jsonRegex := regexp.MustCompile("```json\\s*([\\s\\S]*?)\\s*```")
	matches := jsonRegex.FindStringSubmatch(text)

	var jsonStr string
	if len(matches) >= 2 {
		jsonStr = matches[1]
	} else {
		// JSONブロックがない場合は直接パースを試みる
		jsonStr = text
	}

	var result VerificationResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("failed to parse verification result: %w", err)
	}

	return &result, nil
}

// ExtractEndpoints はコードからAPIエンドポイントを抽出する
func (p *ClaudeProvider) ExtractEndpoints(ctx context.Context, sourceType string, codeContent string) ([]EndpointResult, error) {
	prompt := buildEndpointExtractionPrompt(sourceType, codeContent)

	req := claudeRequest{
		Model:     p.model,
		MaxTokens: 4000,
		Messages: []claudeMessage{
			{Role: "user", Content: prompt},
		},
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", claudeAPIURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var claudeResp claudeResponse
	if err := json.Unmarshal(body, &claudeResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if claudeResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", claudeResp.Error.Message)
	}

	if len(claudeResp.Content) == 0 {
		return nil, fmt.Errorf("empty response from API")
	}

	return parseEndpointResult(claudeResp.Content[0].Text)
}

// buildEndpointExtractionPrompt はエンドポイント抽出用のプロンプトを構築する
func buildEndpointExtractionPrompt(sourceType string, codeContent string) string {
	frameworkHint := ""
	switch sourceType {
	case "express":
		frameworkHint = "Express.js (app.get, app.post, router.get, router.post など)"
	case "fastify":
		frameworkHint = "Fastify (fastify.get, fastify.post など)"
	case "go-echo":
		frameworkHint = "Go Echo (e.GET, e.POST, g.GET など)"
	case "go-gin":
		frameworkHint = "Go Gin (r.GET, r.POST, group.GET など)"
	case "rails":
		frameworkHint = "Ruby on Rails (routes.rb, get/post/resources など)"
	case "django":
		frameworkHint = "Django REST Framework (path, urlpatterns など)"
	case "graphql":
		frameworkHint = "GraphQL (Query, Mutation, type定義)"
	default:
		frameworkHint = "自動検出"
	}

	return fmt.Sprintf(`あなたはAPIエンドポイント抽出の専門家です。
以下のコードからAPIエンドポイントを抽出してください。

## フレームワーク/タイプ
%s

## コード
%s

## 抽出ルール
1. 明確に定義されているエンドポイントのみを抽出してください
2. 推測はしないでください
3. GraphQLの場合は、QueryとMutationを抽出し、methodは "QUERY" または "MUTATION" としてください

## 出力形式
以下のJSON配列形式で出力してください:
%sjson
[
  {
    "method": "GET",
    "path": "/api/users",
    "file": "ファイル名(分かれば)",
    "description": "簡単な説明(あれば)"
  }
]
%s

JSONのみを出力してください。エンドポイントが見つからない場合は空の配列 [] を返してください。`, frameworkHint, codeContent, "```", "```")
}

// parseEndpointResult はClaude APIのレスポンスからエンドポイント結果を抽出する
func parseEndpointResult(text string) ([]EndpointResult, error) {
	// JSONブロックを抽出
	jsonRegex := regexp.MustCompile("```json\\s*([\\s\\S]*?)\\s*```")
	matches := jsonRegex.FindStringSubmatch(text)

	var jsonStr string
	if len(matches) >= 2 {
		jsonStr = matches[1]
	} else {
		// JSONブロックがない場合は直接パースを試みる
		// JSON配列を探す
		arrayRegex := regexp.MustCompile(`\[[\s\S]*\]`)
		arrayMatch := arrayRegex.FindString(text)
		if arrayMatch != "" {
			jsonStr = arrayMatch
		} else {
			jsonStr = text
		}
	}

	var results []EndpointResult
	if err := json.Unmarshal([]byte(jsonStr), &results); err != nil {
		return nil, fmt.Errorf("failed to parse endpoint result: %w", err)
	}

	return results, nil
}
