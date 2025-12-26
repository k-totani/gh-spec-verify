package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadEnvFile(t *testing.T) {
	// テスト用の一時ディレクトリを作成
	tmpDir := t.TempDir()
	envFile := filepath.Join(tmpDir, ".env")

	// テスト用の.envファイルを作成
	content := `# コメント行
TEST_KEY1=value1
TEST_KEY2="quoted value"
TEST_KEY3='single quoted'
EMPTY_LINE_ABOVE=test

# 空行とコメント
TEST_KEY4=value4
`
	if err := os.WriteFile(envFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test env file: %v", err)
	}

	// 元のディレクトリを保存
	origDir, _ := os.Getwd()
	defer os.Chdir(origDir)

	// テストディレクトリに移動
	os.Chdir(tmpDir)

	// 既存の環境変数をクリア
	os.Unsetenv("TEST_KEY1")
	os.Unsetenv("TEST_KEY2")
	os.Unsetenv("TEST_KEY3")
	os.Unsetenv("TEST_KEY4")
	os.Unsetenv("EMPTY_LINE_ABOVE")

	// .envを読み込む
	if err := LoadEnvFile(); err != nil {
		t.Fatalf("LoadEnvFile failed: %v", err)
	}

	// 値を検証
	tests := []struct {
		key      string
		expected string
	}{
		{"TEST_KEY1", "value1"},
		{"TEST_KEY2", "quoted value"},
		{"TEST_KEY3", "single quoted"},
		{"TEST_KEY4", "value4"},
		{"EMPTY_LINE_ABOVE", "test"},
	}

	for _, tt := range tests {
		got := os.Getenv(tt.key)
		if got != tt.expected {
			t.Errorf("os.Getenv(%q) = %q, want %q", tt.key, got, tt.expected)
		}
	}
}

func TestLoadEnvFile_DoesNotOverwriteExisting(t *testing.T) {
	tmpDir := t.TempDir()
	envFile := filepath.Join(tmpDir, ".env")

	content := `EXISTING_KEY=from_file`
	if err := os.WriteFile(envFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test env file: %v", err)
	}

	origDir, _ := os.Getwd()
	defer os.Chdir(origDir)
	os.Chdir(tmpDir)

	// 既存の環境変数をセット
	os.Setenv("EXISTING_KEY", "from_env")
	defer os.Unsetenv("EXISTING_KEY")

	if err := LoadEnvFile(); err != nil {
		t.Fatalf("LoadEnvFile failed: %v", err)
	}

	// 既存の値が保持されていることを確認
	got := os.Getenv("EXISTING_KEY")
	if got != "from_env" {
		t.Errorf("existing env var was overwritten: got %q, want %q", got, "from_env")
	}
}

func TestLoadEnvFile_NoFileExists(t *testing.T) {
	tmpDir := t.TempDir()
	origDir, _ := os.Getwd()
	defer os.Chdir(origDir)
	os.Chdir(tmpDir)

	// ファイルがない場合でもエラーにならないことを確認
	if err := LoadEnvFile(); err != nil {
		t.Errorf("LoadEnvFile should not fail when file doesn't exist: %v", err)
	}
}

func TestGetAPIKeyFromEnv(t *testing.T) {
	// 既存の環境変数を保存してクリア
	savedKeys := map[string]string{
		"SPEC_VERIFY_API_KEY": os.Getenv("SPEC_VERIFY_API_KEY"),
		"ANTHROPIC_API_KEY":   os.Getenv("ANTHROPIC_API_KEY"),
		"OPENAI_API_KEY":      os.Getenv("OPENAI_API_KEY"),
		"GOOGLE_API_KEY":      os.Getenv("GOOGLE_API_KEY"),
	}
	defer func() {
		for k, v := range savedKeys {
			if v != "" {
				os.Setenv(k, v)
			} else {
				os.Unsetenv(k)
			}
		}
	}()

	// すべてクリア
	for k := range savedKeys {
		os.Unsetenv(k)
	}

	tests := []struct {
		name     string
		provider string
		envVars  map[string]string
		expected string
	}{
		{
			name:     "SPEC_VERIFY_API_KEY takes priority",
			provider: "claude",
			envVars: map[string]string{
				"SPEC_VERIFY_API_KEY": "generic-key",
				"ANTHROPIC_API_KEY":   "claude-key",
			},
			expected: "generic-key",
		},
		{
			name:     "claude provider uses ANTHROPIC_API_KEY",
			provider: "claude",
			envVars: map[string]string{
				"ANTHROPIC_API_KEY": "claude-key",
			},
			expected: "claude-key",
		},
		{
			name:     "openai provider uses OPENAI_API_KEY",
			provider: "openai",
			envVars: map[string]string{
				"OPENAI_API_KEY": "openai-key",
			},
			expected: "openai-key",
		},
		{
			name:     "gemini provider uses GOOGLE_API_KEY",
			provider: "gemini",
			envVars: map[string]string{
				"GOOGLE_API_KEY": "google-key",
			},
			expected: "google-key",
		},
		{
			name:     "unknown provider returns empty",
			provider: "unknown",
			envVars:  map[string]string{},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 環境変数をクリア
			for k := range savedKeys {
				os.Unsetenv(k)
			}
			// テスト用の環境変数をセット
			for k, v := range tt.envVars {
				os.Setenv(k, v)
			}

			got := GetAPIKeyFromEnv(tt.provider)
			if got != tt.expected {
				t.Errorf("GetAPIKeyFromEnv(%q) = %q, want %q", tt.provider, got, tt.expected)
			}
		})
	}
}
