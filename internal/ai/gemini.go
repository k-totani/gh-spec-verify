package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const geminiAPIURLTemplate = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s"

// GeminiProvider はGemini APIを使用したプロバイダー
type GeminiProvider struct {
	apiKey string
	model  string
}

// NewGeminiProvider は新しいGeminiProviderを作成する
func NewGeminiProvider(apiKey string) (*GeminiProvider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	return &GeminiProvider{
		apiKey: apiKey,
		model:  "gemini-2.0-flash",
	}, nil
}

// Name はプロバイダー名を返す
func (p *GeminiProvider) Name() string {
	return "gemini"
}

// geminiRequest はGemini APIへのリクエスト
type geminiRequest struct {
	Contents         []geminiContent         `json:"contents"`
	GenerationConfig *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenerationConfig struct {
	MaxOutputTokens int     `json:"maxOutputTokens,omitempty"`
	Temperature     float64 `json:"temperature,omitempty"`
}

// geminiResponse はGemini APIからのレスポンス
type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error,omitempty"`
}

// Verify はSPECとコードの一致度を検証する
func (p *GeminiProvider) Verify(ctx context.Context, specContent string, codeContents map[string]string) (*VerificationResult, error) {
	return p.VerifyWithOptions(ctx, specContent, codeContents, nil)
}

// VerifyWithOptions は検証観点を指定してSPECとコードの一致度を検証する
func (p *GeminiProvider) VerifyWithOptions(ctx context.Context, specContent string, codeContents map[string]string, opts *VerifyOptions) (*VerificationResult, error) {
	var prompt string
	if opts != nil && len(opts.VerificationFocus) > 0 {
		prompt = buildVerificationPromptWithFocus(specContent, codeContents, opts.VerificationFocus)
	} else {
		prompt = buildVerificationPrompt(specContent, codeContents)
	}

	text, err := p.callAPI(ctx, prompt, 2000)
	if err != nil {
		return nil, err
	}

	return parseVerificationResult(text)
}

// ExtractEndpoints はコードからAPIエンドポイント/ページルートを抽出する
func (p *GeminiProvider) ExtractEndpoints(ctx context.Context, opts *ExtractOptions, codeContent string) ([]EndpointResult, error) {
	var prompt string
	if opts.IsUICategory() {
		prompt = buildUIRouteExtractionPrompt(opts.GetSourceType(), codeContent)
	} else {
		prompt = buildEndpointExtractionPrompt(opts.GetSourceType(), codeContent)
	}

	text, err := p.callAPI(ctx, prompt, 4000)
	if err != nil {
		return nil, err
	}

	return parseEndpointResult(text)
}

// callAPI はGemini APIを呼び出す共通関数
func (p *GeminiProvider) callAPI(ctx context.Context, prompt string, maxTokens int) (string, error) {
	req := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
				},
			},
		},
		GenerationConfig: &geminiGenerationConfig{
			MaxOutputTokens: maxTokens,
			Temperature:     0.1,
		},
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	apiURL := fmt.Sprintf(geminiAPIURLTemplate, p.model, p.apiKey)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var geminiResp geminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	if geminiResp.Error != nil {
		return "", fmt.Errorf("API error: %s", geminiResp.Error.Message)
	}

	if len(geminiResp.Candidates) == 0 ||
		len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	return geminiResp.Candidates[0].Content.Parts[0].Text, nil
}
