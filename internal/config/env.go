package config

import (
	"bufio"
	"os"
	"strings"
)

// LoadEnvFile は.envファイルを読み込んで環境変数にセットする
// 既存の環境変数は上書きしない（環境変数が優先）
func LoadEnvFile(paths ...string) error {
	if len(paths) == 0 {
		paths = []string{".env", ".env.local"}
	}

	for _, path := range paths {
		if err := loadEnvFileIfExists(path); err != nil {
			return err
		}
	}
	return nil
}

func loadEnvFileIfExists(path string) error {
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil // ファイルがなければスキップ
	}
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// 空行やコメントをスキップ
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// KEY=VALUE の形式をパース
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		// クォートを除去
		value = strings.Trim(value, `"'`)

		// 既存の環境変数がなければセット
		if os.Getenv(key) == "" {
			os.Setenv(key, value)
		}
	}

	return scanner.Err()
}

// GetAPIKeyFromEnv は環境変数からAPIキーを取得する（優先順位付き）
func GetAPIKeyFromEnv(provider string) string {
	// 汎用キーを最優先
	if key := os.Getenv("SPEC_VERIFY_API_KEY"); key != "" {
		return key
	}

	// プロバイダー固有の環境変数
	switch provider {
	case "claude":
		return os.Getenv("ANTHROPIC_API_KEY")
	case "openai":
		return os.Getenv("OPENAI_API_KEY")
	case "gemini":
		return os.Getenv("GOOGLE_API_KEY")
	}

	return ""
}
